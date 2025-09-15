from flask import Blueprint, request, render_template, jsonify
from utils import allowed_file, extract_formatted_text, get_common_words, search_keywords_in_text, get_context_around_matches, calculate_match_score, mark_task_cancelled, is_task_cancelled, clear_cancelled_task, start_task_timer, check_task_timeout, clear_task_timer, register_active_process, clear_task_processes, terminate_task_processes
import os
import re
import threading
import uuid
import multiprocessing

# 设置多进程启动方法为spawn，解决CUDA fork问题
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，忽略错误
    pass

# 配置multiprocessing，减少resource_tracker警告
import warnings
warnings.filterwarnings("ignore", message="resource_tracker: process died unexpectedly")

# 设置环境变量来减少multiprocessing的资源跟踪
import os
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

keyword_search_bp = Blueprint('keyword_search', __name__)

# 简易进度存储：{task_id: {total:int, current:int, done:bool, message:str}}
SEARCH_PROGRESS = {}
_SEARCH_PROGRESS_LOCK = threading.Lock()

def _init_search_progress(task_id: str, total: int):
    with _SEARCH_PROGRESS_LOCK:
        SEARCH_PROGRESS[task_id] = {'total': total, 'current': 0, 'done': False, 'message': '初始化'}

def _advance_search_progress(task_id: str, msg: str = ''):
    with _SEARCH_PROGRESS_LOCK:
        p = SEARCH_PROGRESS.get(task_id)
        if p:
            p['current'] += 1
            p['message'] = msg

def _finish_search_progress(task_id: str, msg: str = ''):
    with _SEARCH_PROGRESS_LOCK:
        p = SEARCH_PROGRESS.get(task_id)
        if p:
            p['current'] = p['total']
            p['done'] = True
            p['message'] = msg

def _search_file_worker(args):
    """工作进程函数，用于关键词搜索"""
    rel_path, full_path, keywords, search_mode, task_id = args
    try:
        # 提取文件文本
        lines, _, _ = extract_formatted_text(full_path, is_storage_file=True)
        if not lines:
            return None
        
        file_text = '\n'.join(lines)
        
        # 使用通用搜索方法
        has_match, matches, positions = search_keywords_in_text(file_text, keywords, search_mode)
        
        if has_match:
            # 计算匹配分数
            match_scores = calculate_match_score(matches, len(file_text), positions)
            
            # 获取匹配的上下文
            contexts = get_context_around_matches(file_text, positions, context_size=200)
            
            return {
                'file_path': rel_path,
                'full_path': full_path,
                'matches': matches,
                'match_count': match_scores['match_count'],
                'match_ratio': match_scores['match_ratio'],
                'density_score': match_scores['density_score'],
                'coverage_score': match_scores['coverage_score'],
                'contexts': contexts[:5],  # 限制上下文数量
                'file_size': len(file_text),
                'search_mode': search_mode
            }
        else:
            return None
            
    except Exception as e:
        return None

def _search_batch_worker(args):
    """批量搜索工作进程函数"""
    batch_files, keywords, search_mode, task_id = args
    results = []
    
    try:
        print(f"工作进程开始处理 {len(batch_files)} 个文件...")
        
        for rel_path, full_path in batch_files:
            # 检查任务是否被取消
            with _SEARCH_PROGRESS_LOCK:
                if task_id in SEARCH_PROGRESS and SEARCH_PROGRESS.get(task_id, {}).get('cancelled', False):
                    print(f"搜索任务 {task_id} 已被取消，停止处理")
                    return results
            
            # 检查任务是否超时
            if check_task_timeout(task_id):
                print(f"搜索任务 {task_id} 已超时，停止处理")
                return results
            try:
                # 检查文件大小，避免处理过大文件
                file_size = os.path.getsize(full_path)
                if file_size > 50 * 1024 * 1024:  # 50MB
                    print(f"跳过过大文件 {rel_path} ({file_size/1024/1024:.1f}MB)")
                    continue
                
                # 直接提取文件文本
                lines, _, _ = extract_formatted_text(full_path, is_storage_file=True, task_id=task_id)
                if not lines:
                    continue
                
                file_text = '\n'.join(lines)
                
                # 使用通用搜索方法
                has_match, matches, positions = search_keywords_in_text(file_text, keywords, search_mode)
                
                if has_match:
                    # 计算匹配分数
                    match_scores = calculate_match_score(matches, len(file_text), positions)
                    
                    # 获取匹配的上下文
                    contexts = get_context_around_matches(file_text, positions, context_size=200)
                    
                    results.append({
                        'file_path': rel_path,
                        'full_path': full_path,
                        'matches': matches,
                        'match_count': match_scores['match_count'],
                        'match_ratio': match_scores['match_ratio'],
                        'density_score': match_scores['density_score'],
                        'coverage_score': match_scores['coverage_score'],
                        'contexts': contexts[:5],  # 限制上下文数量
                        'file_size': len(file_text),
                        'search_mode': search_mode
                    })
            except Exception as e:
                print(f"处理文件 {rel_path} 时出错: {e}")
                continue
                
    except Exception as e:
        print(f"批量处理出错: {e}")
    
    return results

@keyword_search_bp.route('/progress/<task_id>', methods=['GET'])
def get_search_progress(task_id):
    with _SEARCH_PROGRESS_LOCK:
        data = SEARCH_PROGRESS.get(task_id)
        if not data:
            return jsonify({'total': 1, 'current': 0, 'done': False, 'message': '未找到任务'}), 200
        return jsonify(data), 200

@keyword_search_bp.route('/cancel/<task_id>', methods=['POST'])
def cancel_search_task(task_id):
    """取消指定的搜索任务"""
    # 无论任务是否已开始，都标记为取消
    mark_task_cancelled(task_id)
    
    with _SEARCH_PROGRESS_LOCK:
        if task_id in SEARCH_PROGRESS:
            # 标记任务为已取消
            SEARCH_PROGRESS[task_id]['cancelled'] = True
            SEARCH_PROGRESS[task_id]['done'] = True
            SEARCH_PROGRESS[task_id]['message'] = '搜索任务已被用户取消'
            print(f"搜索任务 {task_id} 已被取消")
            return jsonify({'status': 'cancelled', 'message': '搜索任务已取消'}), 200
        else:
            # 即使任务还没有开始，也标记为取消
            print(f"搜索任务 {task_id} 在开始前已被取消")
            return jsonify({'status': 'cancelled', 'message': '搜索任务已取消'}), 200

@keyword_search_bp.route('/storage_paths', methods=['GET'])
def get_storage_paths():
    """获取可用的存储库路径列表"""
    storage_paths = []
    
    # 只扫描 storage 目录及其子目录
    if os.path.exists('storage') and os.path.isdir('storage'):
        current_dir = os.getcwd()
        storage_base_path = os.path.join(current_dir, 'storage')
        
        # 添加 storage 根目录
        storage_paths.append({
            'path': 'storage',
            'name': 'storage (根目录)',
            'full_path': storage_base_path
        })
        
        # 扫描 storage 目录下的所有子目录
        for root, dirs, files in os.walk(storage_base_path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                # 检查子目录中是否有支持的文件
                has_files = False
                for sub_root, _, sub_files in os.walk(dir_path):
                    for file in sub_files:
                        if allowed_file(file):
                            has_files = True
                            break
                    if has_files:
                        break
                
                if has_files:
                    # 计算相对于 storage 的路径
                    rel_path = os.path.relpath(dir_path, storage_base_path)
                    storage_paths.append({
                        'path': f'storage/{rel_path}',
                        'name': f'storage/{rel_path}',
                        'full_path': dir_path
                    })
    
    return jsonify({
        'storage_paths': storage_paths
    }), 200

@keyword_search_bp.route('/file_types', methods=['GET'])
def get_storage_file_types():
    """获取存储库中所有文件类型"""
    # 获取路径参数，默认为 'storage'
    storage_path = request.args.get('path', 'storage')
    
    # 验证路径安全性
    if not os.path.exists(storage_path) or not os.path.isdir(storage_path):
        return jsonify({
            'error': f'路径不存在或不是目录: {storage_path}',
            'file_types': [],
            'file_count_by_type': {}
        }), 400
    
    file_types = set()
    file_count_by_type = {}
    
    # 递归遍历指定目录及其子目录
    for root, _, files in os.walk(storage_path):
        for file in files:
            if allowed_file(file):
                ext = file.rsplit('.', 1)[-1].lower() if '.' in file else 'no_extension'
                file_types.add(ext)
                file_count_by_type[ext] = file_count_by_type.get(ext, 0) + 1
    
    # 转换为列表并排序
    file_types_list = sorted(list(file_types))
    
    return jsonify({
        'file_types': file_types_list,
        'file_count_by_type': file_count_by_type,
        'storage_path': storage_path
    }), 200

@keyword_search_bp.route('/', methods=['GET', 'POST'])
def keyword_search():
    # 在函数开始时就设置环境变量，抑制子进程中的初始化日志
    os.environ['SUPPRESS_INIT_LOGS'] = '1'
    
    print("关键词搜索函数被调用")
    print(f"请求方法: {request.method}")
    results = []
    error = None

    if request.method == 'POST':
        # 获取用户输入
        keywords = request.form.get('keywords', '').strip()
        selected_storage_path = request.form.get('selected_storage_path', 'storage')
        selected_types = request.form.getlist('selected_file_types')
        search_mode = request.form.get('search_mode', 'exact')  # exact, fuzzy, regex
        
        if not keywords:
            error = '请输入关键词'
        else:
            # 任务ID（用于前端真实进度）
            task_id = request.form.get('task_id') or str(uuid.uuid4())
            
            # 等待一小段时间再检查任务是否被取消，避免前端页面跳转时的误触发
            import time
            time.sleep(1.0)  # 等待1秒钟，给前端页面跳转足够的时间
            
            # 检查任务是否已被取消
            if is_task_cancelled(task_id):
                print(f"任务 {task_id} 在开始前已被取消")
                # 清理取消状态，避免影响后续任务
                clear_cancelled_task(task_id)
                # 返回正常的结果页面，但显示取消状态
                return render_template('keyword_search.html', 
                                     results=[], 
                                     error=None, 
                                     task_cancelled=True,
                                     keywords=keywords,
                                     search_mode=search_mode)
            
            # 开始任务计时
            start_task_timer(task_id)
            
            # 记录任务开始时的进程列表
            initial_processes = set()
            try:
                import psutil
                for proc in psutil.process_iter(['pid']):
                    initial_processes.add(proc.info['pid'])
                print(f"任务开始时记录了 {len(initial_processes)} 个进程")
            except Exception as e:
                print(f"记录初始进程时出错: {e}")
            
            try:
                # 获取要搜索的文件列表
                search_files = []
                
                # 递归遍历指定目录及其子目录
                for root, _, files in os.walk(selected_storage_path):
                    for file in files:
                        if allowed_file(file):
                            rel_path = os.path.relpath(os.path.join(root, file), selected_storage_path)
                            full_path = os.path.join(root, file)
                            
                            # 如果指定了文件类型，进行筛选
                            if selected_types:
                                file_ext = file.rsplit('.', 1)[-1].lower() if '.' in file else 'no_extension'
                                if file_ext in selected_types:
                                    search_files.append((rel_path, full_path))
                            else:
                                search_files.append((rel_path, full_path))
                
                if not search_files:
                    error = '未找到符合条件的文件'
                else:
                    print(f"开始搜索 {len(search_files)} 个文件中的关键词: {keywords}")
                    
                    # 初始化进度
                    _init_search_progress(task_id, len(search_files))
                    
                    # 并行搜索文件
                    from concurrent.futures import ThreadPoolExecutor
                    import multiprocessing
                    
                    max_workers = min(multiprocessing.cpu_count(), 16)  # 增加进程数
                    print(f"使用 {max_workers} 个进程进行关键词搜索")
                    
                    # 将文件分批处理，每个进程处理一批文件
                    batch_size = max(1, len(search_files) // max_workers)
                    file_batches = [search_files[i:i + batch_size] for i in range(0, len(search_files), batch_size)]
                    
                    # 准备批量参数
                    batch_args = [(batch, keywords, search_mode, task_id) for batch in file_batches]
                    
                    from concurrent.futures import ProcessPoolExecutor
                    # 使用更保守的进程数，减少资源竞争
                    max_workers = min(multiprocessing.cpu_count(), 8)  # 减少到8个进程
                    print(f"使用 {max_workers} 个进程进行关键词搜索")
                    executor = ProcessPoolExecutor(max_workers=max_workers)
                    
                    try:
                        # 提交任务并获取Future对象
                        futures = []
                        for batch_arg in batch_args:
                            future = executor.submit(_search_batch_worker, batch_arg)
                            futures.append(future)
                        
                        # 等待所有任务完成，但定期检查取消状态
                        batch_results = []
                        for future in futures:
                            if is_task_cancelled(task_id):
                                print(f"任务 {task_id} 在等待结果时被取消")
                                # 取消所有未完成的任务
                                for f in futures:
                                    f.cancel()
                                break
                            try:
                                result = future.result(timeout=120)  # 120秒超时，给足够时间处理大文件
                                if result:
                                    batch_results.append(result)
                            except Exception as e:
                                if not is_task_cancelled(task_id):
                                    print(f"批次处理出错: {type(e).__name__}: {e}")
                                    import traceback
                                    print(f"详细错误信息: {traceback.format_exc()}")
                                
                    finally:
                        # 确保进程池被正确关闭
                        try:
                            if is_task_cancelled(task_id):
                                print(f"任务 {task_id} 已取消，强制关闭进程池")
                                executor.shutdown(wait=False, cancel_futures=True)
                            else:
                                print(f"任务 {task_id} 正常关闭进程池")
                                executor.shutdown(wait=True)
                        except Exception as e:
                            print(f"关闭进程池时出错: {e}")
                    
                    # 合并所有批次的结果
                    search_results = []
                    for i, batch_result in enumerate(batch_results):
                        print(f"批次 {i+1} 结果: {len(batch_result) if batch_result else 0} 个匹配")
                        if batch_result:
                            search_results.extend(batch_result)
                    
                    # 过滤掉None结果
                    results = [r for r in search_results if r is not None]
                    print(f"总共找到 {len(results)} 个匹配文件")
                    
                    # 按综合分数排序（匹配度 + 密度分数）
                    results.sort(key=lambda x: x['match_ratio'] + x['density_score'], reverse=True)
                    
                    print(f"搜索完成：找到 {len(results)} 个匹配文件")
                    _finish_search_progress(task_id, f'搜索完成，找到 {len(results)} 个匹配文件')
                    
                    # 立即强制清理所有相关进程
                    print("开始强制清理所有相关进程...")
                    try:
                        import psutil
                        current_pid = os.getpid()
                        killed_count = 0
                        
                        # 获取当前所有进程
                        current_processes = set()
                        for proc in psutil.process_iter(['pid']):
                            current_processes.add(proc.info['pid'])
                        
                        # 找出新创建的进程
                        new_processes = current_processes - initial_processes
                        print(f"发现 {len(new_processes)} 个新创建的进程")
                        
                        # 杀死所有新创建的Python进程
                        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                            try:
                                if proc.info['pid'] in new_processes and proc.info['name'] == 'python' and proc.info['pid'] != current_pid:
                                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                                    # 检查是否是multiprocessing相关进程
                                    if 'multiprocessing' in cmdline or 'spawn_main' in cmdline or 'storage_compare' in cmdline or 'keyword_search' in cmdline:
                                        print(f"关键词搜索：发现相关进程 {proc.info['pid']}，命令行: {cmdline[:100]}...")
                                        proc.terminate()
                                        killed_count += 1
                                        # 等待进程终止
                                        try:
                                            proc.wait(timeout=2)
                                        except psutil.TimeoutExpired:
                                            print(f"进程 {proc.info['pid']} 未响应terminate，强制杀死")
                                            proc.kill()
                            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                                pass
                        print(f"已终止 {killed_count} 个新创建的进程")
                    except Exception as e:
                        print(f"强制清理进程时出错: {e}")
                    
                    # 清理取消状态
                    clear_cancelled_task(task_id)
                    # 清除任务计时
                    clear_task_timer(task_id)
                    # 强制终止所有相关进程
                    terminate_task_processes(task_id)
                    # 清除进程记录
                    clear_task_processes(task_id)
                    # 额外清理所有multiprocessing进程
                    from utils import kill_all_multiprocessing_processes, force_kill_all_python_processes
                    kill_all_multiprocessing_processes()
                    # 强制清理所有Python进程
                    force_kill_all_python_processes()
                    
            except Exception as e:
                error = f"搜索过程中出错: {str(e)}"
                print(f"搜索错误: {e}")
                _finish_search_progress(task_id, f'搜索失败: {error}')
                
                # 立即强制清理所有相关进程
                print("异常处理：开始强制清理所有相关进程...")
                try:
                    import psutil
                    current_pid = os.getpid()
                    killed_count = 0
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            if proc.info['name'] == 'python' and proc.info['pid'] != current_pid:
                                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                                if 'keyword_search' in cmdline or 'storage_compare' in cmdline or 'multiprocessing' in cmdline:
                                    print(f"异常处理：发现相关进程 {proc.info['pid']}，强制终止")
                                    proc.terminate()
                                    killed_count += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    print(f"异常处理：已终止 {killed_count} 个相关进程")
                except Exception as cleanup_e:
                    print(f"异常处理：强制清理进程时出错: {cleanup_e}")
                
                # 清理取消状态
                clear_cancelled_task(task_id)
                # 清除任务计时
                clear_task_timer(task_id)
                # 强制终止所有相关进程
                terminate_task_processes(task_id)
                # 清除进程记录
                clear_task_processes(task_id)
                # 额外清理所有multiprocessing进程
                from utils import kill_all_multiprocessing_processes, force_kill_all_python_processes
                kill_all_multiprocessing_processes()
                # 强制清理所有Python进程
                force_kill_all_python_processes()

    # 清理环境变量
    os.environ.pop('SUPPRESS_INIT_LOGS', None)
    
    if request.method == 'POST':
        # POST请求：处理搜索结果
        if results:
            return render_template('keyword_search.html', results=results, keywords=keywords, error=error)
        else:
            return render_template('index.html', results=None, error=error)
    else:
        # GET请求：用户直接访问或刷新结果页面，重定向到首页
        from flask import redirect, url_for
        print("关键词搜索GET请求，重定向到首页")
        return redirect(url_for('index'))
