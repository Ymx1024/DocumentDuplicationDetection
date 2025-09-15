from flask import Blueprint, request, render_template, send_from_directory, jsonify
from utils import allowed_file, extract_formatted_text, compute_similarity, get_common_words, compute_differences, \
    get_similarity_reason, load_similarities, save_similarities, mark_task_cancelled, is_task_cancelled, \
    clear_cancelled_task, start_task_timer, check_task_timeout, clear_task_timer, terminate_task_processes, clear_task_processes, \
    get_user_id_from_request, register_user_session, register_task_to_user, update_system_resources
import os
import json
import uuid
import threading
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

storage_compare_bp = Blueprint('storage_compare', __name__)

# 简易进度存储：{task_id: {total:int, current:int, done:bool, message:str}}
PROGRESS = {}
_PROGRESS_LOCK = threading.Lock()

def _process_storage_file_worker(args):
    """工作进程函数，用于存储库比较"""
    rel_path, storage_path, uploaded_text, uploaded_filename, uploaded_lines, temp_file_path, task_id = args
    try:
        storage_lines, storage_html, storage_temp_path = extract_formatted_text(storage_path, is_storage_file=True)
        if not storage_lines:
            return {'filename': rel_path, 'error': '无法提取文本'}
        storage_text = '\n'.join(storage_lines)
        similarity = compute_similarity(uploaded_text, storage_text)
        common_words = get_common_words(uploaded_text, storage_text)
        # 仅保留前端所需的相似度与高频词/原因，移除逐行差异
        similarity_reason = get_similarity_reason(uploaded_text, storage_text, common_words, similarity)
        return {
            'filename': rel_path,
            'similarity': round(similarity * 100, 2),
            'common_words': common_words,
            'similarity_reason': similarity_reason,
            'uploaded_filename': uploaded_filename,
            'uploaded_lines': uploaded_lines,
            'storage_lines': storage_lines,
            'temp_file_path': temp_file_path,
            'storage_temp_path': storage_temp_path
        }
    except Exception as e:
        return {'filename': rel_path, 'error': str(e)}

def _process_storage_batch_worker(args):
    """批量处理存储库文件的工作进程函数"""
    batch_files, uploaded_text, uploaded_filename, uploaded_lines, temp_file_path, task_id = args
    results = []
    
    try:
        for rel_path, storage_path in batch_files:
            # 检查任务是否被取消（更频繁的检查）
            with _PROGRESS_LOCK:
                if task_id in PROGRESS and PROGRESS.get(task_id, {}).get('cancelled', False):
                    print(f"任务 {task_id} 已被取消，停止处理")
                    return results
            
            # 额外检查全局取消状态
            from utils import is_task_cancelled
            if is_task_cancelled(task_id):
                print(f"任务 {task_id} 已被全局取消，停止处理")
                return results
            
            # 在开始处理每个文件前再次检查
            if is_task_cancelled(task_id):
                print(f"任务 {task_id} 在开始处理文件 {rel_path} 前已被取消")
                return results
                
            try:
                # 在开始处理文件前再次检查任务是否被取消
                if is_task_cancelled(task_id):
                    print(f"任务 {task_id} 在开始处理文件 {rel_path} 前已被取消")
                    return results
                
                storage_lines, storage_html, storage_temp_path = extract_formatted_text(storage_path, is_storage_file=True, task_id=task_id)
                if not storage_lines:
                    results.append({'filename': rel_path, 'error': '无法提取文本'})
                    continue
                
                # 在文本提取后再次检查任务是否被取消
                if is_task_cancelled(task_id):
                    print(f"任务 {task_id} 在文本提取后已被取消")
                    return results
                    
                storage_text = '\n'.join(storage_lines)
                
                # 在相似度计算前再次检查任务是否被取消
                if is_task_cancelled(task_id):
                    print(f"任务 {task_id} 在相似度计算前已被取消")
                    return results
                
                similarity = compute_similarity(uploaded_text, storage_text)
                
                # 在相似度计算后再次检查任务是否被取消
                if is_task_cancelled(task_id):
                    print(f"任务 {task_id} 在相似度计算后已被取消")
                    return results
                
                common_words = get_common_words(uploaded_text, storage_text)
                # 仅保留前端所需的相似度与高频词/原因，移除逐行差异
                similarity_reason = get_similarity_reason(uploaded_text, storage_text, common_words, similarity)
                
                # 清理GPU内存
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                
                results.append({
                    'filename': rel_path,
                    'similarity': round(similarity * 100, 2),
                    'common_words': common_words,
                    'similarity_reason': similarity_reason,
                    'uploaded_filename': uploaded_filename,
                    'uploaded_lines': uploaded_lines,
                    'storage_lines': storage_lines,
                    'temp_file_path': temp_file_path,
                    'storage_temp_path': storage_temp_path
                })
                
                # 处理完一个文件后再次检查任务是否被取消
                if is_task_cancelled(task_id):
                    print(f"任务 {task_id} 在处理文件 {rel_path} 后已被取消，停止处理")
                    return results
                    
            except Exception as e:
                print(f"处理文件 {rel_path} 时出错: {type(e).__name__}: {e}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                results.append({'filename': rel_path, 'error': str(e)})
                
    except Exception as e:
        print(f"批次处理工作进程出错: {type(e).__name__}: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        # 返回一个包含错误信息的结果
        return [{'filename': 'batch_error', 'error': str(e)}]
    
    return results

def _init_progress(task_id: str, total: int):
    with _PROGRESS_LOCK:
        PROGRESS[task_id] = {'total': total, 'current': 0, 'done': False, 'message': '初始化'}

def _advance_progress(task_id: str, msg: str = ''):
    with _PROGRESS_LOCK:
        p = PROGRESS.get(task_id)
        if p:
            p['current'] += 1
            p['message'] = msg

def _finish_progress(task_id: str, msg: str = ''):
    with _PROGRESS_LOCK:
        p = PROGRESS.get(task_id)
        if p:
            p['current'] = p['total']
            p['done'] = True
            p['message'] = msg

def _check_and_cleanup_process_pool():
    """检查并清理可能存在的残留进程池进程"""
    try:
        import psutil
        current_pid = os.getpid()
        cleanup_count = 0
        
        # 查找所有Python进程
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
            try:
                if proc.info['name'] == 'python' and proc.info['pid'] != current_pid:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    # 检查是否是multiprocessing相关进程
                    if any(keyword in cmdline for keyword in ['multiprocessing', 'spawn_main', 'storage_compare', 'keyword_search']):
                        print(f"发现残留进程 {proc.info['pid']}，命令行: {cmdline[:100]}...")
                        # 检查进程是否还在运行
                        if proc.is_running():
                            print(f"终止残留进程 {proc.info['pid']}")
                            proc.terminate()
                            cleanup_count += 1
                            # 等待进程终止
                            try:
                                proc.wait(timeout=2)
                            except psutil.TimeoutExpired:
                                print(f"进程 {proc.info['pid']} 未响应terminate，强制杀死")
                                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
        
        if cleanup_count > 0:
            print(f"清理了 {cleanup_count} 个残留进程")
        else:
            print("未发现残留进程")
            
    except Exception as e:
        print(f"检查残留进程时出错: {e}")

@storage_compare_bp.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    with _PROGRESS_LOCK:
        data = PROGRESS.get(task_id)
        if not data:
            return jsonify({'total': 1, 'current': 0, 'done': False, 'message': '未找到任务'}), 200
        return jsonify(data), 200

@storage_compare_bp.route('/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """取消指定的任务"""
    # 无论任务是否已开始，都标记为取消
    mark_task_cancelled(task_id)
    
    with _PROGRESS_LOCK:
        if task_id in PROGRESS:
            # 标记任务为已取消
            PROGRESS[task_id]['cancelled'] = True
            PROGRESS[task_id]['done'] = True
            PROGRESS[task_id]['message'] = '任务已被用户取消'
            print(f"任务 {task_id} 已被取消")
            return jsonify({'status': 'cancelled', 'message': '任务已取消'}), 200
        else:
            # 即使任务还没有开始，也标记为取消
            print(f"任务 {task_id} 在开始前已被取消")
            return jsonify({'status': 'cancelled', 'message': '任务已取消'}), 200


@storage_compare_bp.route('/file_types', methods=['GET'])
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


@storage_compare_bp.route('/storage_paths', methods=['GET'])
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

@storage_compare_bp.route('/', methods=['GET', 'POST'])
def storage_compare():
    # 在函数开始时就设置环境变量，抑制子进程中的初始化日志
    os.environ['SUPPRESS_INIT_LOGS'] = '1'
    
    # 获取用户ID并注册会话
    user_id = get_user_id_from_request(request)
    register_user_session(user_id, request)
    
    print(f"存储库比较函数被调用 - 用户: {user_id}")
    print(f"请求方法: {request.method}")
    results = []
    uploaded_file = None
    error = None
    temp_file_path = None  # 初始化 temp_file_path，避免未定义

    if request.method == 'POST':
        if 'uploaded_file' not in request.files:
            error = '请上传文件'
        else:
            uploaded_file = request.files['uploaded_file']
            if uploaded_file.filename == '':
                error = '未选择文件'
            elif not allowed_file(uploaded_file.filename):
                error = '不支持的文件类型'
            else:
                # 任务ID（用于前端真实进度）
                task_id = request.form.get('task_id') or str(uuid.uuid4())
                print(f"使用任务ID: {task_id}")
                
                # 将任务注册到用户
                register_task_to_user(task_id, user_id)
                
                # 开始任务计时
                start_task_timer(task_id)
                
                # 记录任务开始时的进程列表
                initial_processes = set()
                try:
                    import psutil
                    for proc in psutil.process_iter(['pid']):
                        initial_processes.add(proc.info['pid'])
                    print(f"存储库比较任务开始时记录了 {len(initial_processes)} 个进程")
                except Exception as e:
                    print(f"记录初始进程时出错: {e}")
                
                # 等待一小段时间再检查任务是否被取消，避免前端页面跳转时的误触发
                import time
                time.sleep(1.0)  # 等待1秒钟，给前端页面跳转足够的时间
                
                # 检查任务是否已被取消
                if is_task_cancelled(task_id):
                    print(f"任务 {task_id} 在开始前已被取消")
                    # 清理取消状态，避免影响后续任务
                    clear_cancelled_task(task_id)
                    return render_template('index.html', results=None, uploaded_file=uploaded_file, error='任务已被取消')
                
                # 在文件上传后再次检查任务是否被取消
                if is_task_cancelled(task_id):
                    print(f"任务 {task_id} 在文件上传后已被取消")
                    clear_cancelled_task(task_id)
                    return render_template('index.html', results=None, uploaded_file=uploaded_file, error='任务已被取消')
                
                # 使用 UUID 生成唯一临时文件名
                ext = uploaded_file.filename.rsplit('.', 1)[1].lower()
                temp_filename = f"{uuid.uuid4()}.{ext}"
                upload_path = os.path.join('Uploads', temp_filename)
                uploaded_file.save(upload_path)
                try:
                    # 提取上传文件，标记为非存储库文件
                    uploaded_lines, uploaded_html, temp_file_path = extract_formatted_text(upload_path,
                                                                                           is_storage_file=False)
                    if not uploaded_lines:
                        error = '无法提取上传文件文本'
                    else:
                        similarities = load_similarities()
                        storage_files = []
                        
                        # 获取用户选择的存储库路径
                        selected_storage_path = request.form.get('selected_storage_path', 'storage')
                        
                        # 递归遍历指定目录及其子目录
                        for root, _, files in os.walk(selected_storage_path):
                            for file in files:
                                if allowed_file(file):
                                    rel_path = os.path.relpath(os.path.join(root, file), selected_storage_path)
                                    storage_files.append((rel_path, os.path.join(root, file)))

                        # 根据用户选择的文件类型进行筛选
                        selected_types = request.form.getlist('selected_file_types')
                        if selected_types:
                            print(f"用户选择的文件类型: {selected_types}")
                            before = len(storage_files)
                            storage_files = [(rel, p) for (rel, p) in storage_files 
                                           if rel.rsplit('.', 1)[-1].lower() in selected_types]
                            print(f"按类型筛选: {before} -> {len(storage_files)}")
                        else:
                            print("未选择任何文件类型，将比对所有文件")

                        # 打印存储库文件列表
                        if storage_files:
                            print(f"发现存储库文件 {len(storage_files)} 个：")
                            for rel, _p in storage_files:
                                print(f"  - {rel}")
                        else:
                            print("存储库中未找到可比对文件。")

                        if not storage_files:
                            error = '存储库中无文件'
                        else:
                            uploaded_text = '\n'.join(uploaded_lines)
                            print(f"开始并行处理 {len(storage_files)} 个存储库文件...")
                            
                            # 初始化进度
                            _init_progress(task_id, len(storage_files))
                            
                            # 再次检查任务是否被取消（在开始处理前）
                            if is_task_cancelled(task_id):
                                print(f"任务 {task_id} 在开始处理前已被取消")
                                _finish_progress(task_id, '任务已取消')
                                return render_template('index.html', results=None, uploaded_file=uploaded_file, error='任务已被取消')
                            
                            # 并行处理存储库文件
                            from concurrent.futures import ThreadPoolExecutor
                            import multiprocessing
                            
                            max_workers = min(multiprocessing.cpu_count(), 16)  # 增加进程数
                            print(f"使用 {max_workers} 个进程进行并行处理")
                            
                            # 将文件分批处理，每个进程处理一批文件
                            batch_size = max(1, len(storage_files) // max_workers)
                            file_batches = [storage_files[i:i + batch_size] for i in range(0, len(storage_files), batch_size)]
                            
                            # 准备批量参数（只传递可序列化的数据）
                            batch_args = [(batch, uploaded_text, uploaded_file.filename, uploaded_lines, temp_file_path, task_id) 
                                        for batch in file_batches]
                            
                            # 监控CPU和GPU使用情况
                            from utils import monitor_cpu_usage, monitor_gpu_memory, cleanup_gpu_memory
                            print("开始处理前的系统状态:")
                            monitor_cpu_usage()
                            monitor_gpu_memory()
                            
                            # 清理GPU内存
                            cleanup_gpu_memory()
                            
                            # 使用进程池而不是线程池，避免GIL限制
                            from concurrent.futures import ProcessPoolExecutor
                            
                            batch_results = []
                            
                            # 检查任务是否被取消（在开始进程池前）
                            if is_task_cancelled(task_id):
                                print(f"任务 {task_id} 在开始进程池前已被取消")
                                _finish_progress(task_id, '任务已取消')
                                return render_template('index.html', results=None, uploaded_file=uploaded_file, error='任务已被取消')
                            
                            # 使用更简单可靠的进程池执行方式
                            executor = None
                            try:
                                # 在创建进程池前检查任务是否被取消
                                if is_task_cancelled(task_id):
                                    print(f"任务 {task_id} 在创建进程池前已被取消")
                                    _finish_progress(task_id, '任务已取消')
                                    return render_template('index.html', results=None, uploaded_file=uploaded_file, error='任务已被取消')
                                
                                # 在创建新进程池前，检查并清理可能存在的残留进程
                                print(f"任务 {task_id} 创建进程池前检查残留进程...")
                                _check_and_cleanup_process_pool()
                                
                                executor = ProcessPoolExecutor(max_workers=max_workers)
                                print(f"任务 {task_id} 进程池已创建，工作进程数: {max_workers}")
                                
                                # 提交所有任务并等待结果
                                future_to_batch = {executor.submit(_process_storage_batch_worker, args): args for args in batch_args}
                                print(f"任务 {task_id} 已提交 {len(future_to_batch)} 个批次任务")
                                
                                # 逐个获取结果，允许及时响应取消信号
                                completed_count = 0
                                for future in future_to_batch:
                                    # 检查任务是否被取消
                                    if is_task_cancelled(task_id):
                                        print(f"任务 {task_id} 在进程池执行过程中已被取消，立即终止进程池")
                                        # 取消所有未完成的任务
                                        for f in future_to_batch:
                                            if not f.done():
                                                f.cancel()
                                        # 立即关闭进程池
                                        executor.shutdown(wait=False, cancel_futures=True)
                                        print(f"任务 {task_id} 进程池已强制关闭")
                                        break
                                    
                                    try:
                                        # 使用更长的超时时间，允许大文件处理
                                        result = future.result(timeout=120)  # 增加超时时间到120秒
                                        if result:
                                            batch_results.append(result)
                                            completed_count += 1
                                            print(f"任务 {task_id} 批次 {completed_count}/{len(future_to_batch)} 完成")
                                    except Exception as e:
                                        print(f"批次处理出错: {type(e).__name__}: {e}")
                                        import traceback
                                        print(f"详细错误信息: {traceback.format_exc()}")
                                        
                                        # 检查是否因为任务取消而中断
                                        if is_task_cancelled(task_id):
                                            print(f"任务 {task_id} 已取消，立即终止进程池")
                                            executor.shutdown(wait=False, cancel_futures=True)
                                            break
                                        
                                        # 如果是超时错误，记录详细信息
                                        if isinstance(e, TimeoutError):
                                            print(f"批次处理超时：某个文件处理时间超过120秒")
                                            print(f"建议：检查文件大小，大文件可能需要更长时间处理")
                                        
                                        # 如果是进程池损坏错误，记录详细信息
                                        if "BrokenProcessPool" in str(type(e)):
                                            print(f"进程池损坏：{e}")
                                            print(f"这通常是因为之前的进程池没有正确清理导致的")
                                            print(f"将尝试强制清理所有相关进程")
                                            # 强制清理所有相关进程
                                            from utils import force_kill_all_pt_main_thread_processes_safe
                                            force_kill_all_pt_main_thread_processes_safe()
                                            break
                                            
                            except Exception as e:
                                print(f"进程池执行出错: {e}")
                                if is_task_cancelled(task_id):
                                    print(f"任务 {task_id} 已取消，停止进程池执行")
                                else:
                                    # 如果不是因为取消而失败，尝试获取已完成的结果
                                    print("尝试获取已完成的结果...")
                                    for future in future_to_batch:
                                        try:
                                            if future.done() and not future.cancelled():
                                                result = future.result()
                                                if result:
                                                    batch_results.append(result)
                                        except Exception as e:
                                            print(f"获取结果时出错: {e}")
                            finally:
                                # 确保进程池被正确关闭
                                if executor:
                                    try:
                                        if is_task_cancelled(task_id):
                                            print(f"任务 {task_id} 已取消，强制关闭进程池")
                                            # 先尝试正常关闭，给进程一些时间清理
                                            try:
                                                executor.shutdown(wait=True, cancel_futures=True)
                                                print(f"任务 {task_id} 进程池正常关闭完成")
                                            except Exception as shutdown_error:
                                                print(f"正常关闭进程池失败: {shutdown_error}")
                                                # 如果正常关闭失败，强制关闭
                                                executor.shutdown(wait=False, cancel_futures=True)
                                                print(f"任务 {task_id} 进程池强制关闭完成")
                                            
                                            # 立即强制清理所有相关进程
                                            from utils import force_cleanup_user_processes, force_kill_pt_main_thread_processes, force_kill_all_pt_main_thread_processes_safe
                                            user_id = get_user_id_from_request(request)
                                            force_cleanup_user_processes(user_id)
                                            force_kill_pt_main_thread_processes(user_id)
                                            # 额外清理：杀死所有pt_main_thread进程（除了主服务进程）
                                            force_kill_all_pt_main_thread_processes_safe()
                                        else:
                                            print(f"任务 {task_id} 正常关闭进程池")
                                            # 正常关闭时使用温和的清理策略
                                            executor.shutdown(wait=True)
                                            print(f"任务 {task_id} 进程池正常关闭完成")
                                        print(f"任务 {task_id} 进程池已关闭")
                                    except Exception as e:
                                        print(f"关闭进程池时出错: {e}")
                                        # 出错时也要进行进程清理
                                        try:
                                            from utils import force_kill_all_pt_main_thread_processes_safe
                                            force_kill_all_pt_main_thread_processes_safe()
                                        except:
                                            pass
                            
                            # 检查任务是否被取消，如果被取消则直接返回
                            if is_task_cancelled(task_id):
                                print(f"任务 {task_id} 在合并结果前已被取消，直接返回")
                                _finish_progress(task_id, '任务已取消')
                                return render_template('index.html', results=None, uploaded_file=uploaded_file, 
                                                     error='任务已被取消', 
                                                     error_details='用户取消了任务处理',
                                                     suggestions=['请重新上传文件进行处理', '如果文件较大，请耐心等待'])
                            
                            # 合并所有批次的结果
                            file_results = []
                            timeout_count = 0
                            for batch_result in batch_results:
                                if batch_result:
                                    file_results.extend(batch_result)
                            
                            # 检查是否有超时错误
                            if len(file_results) == 0 and len(batch_args) > 0:
                                print("所有批次都处理失败，可能是超时或内存问题")
                                _finish_progress(task_id, '处理失败')
                                return render_template('index.html', results=None, uploaded_file=uploaded_file, 
                                                     error='文件处理失败', 
                                                     error_details='所有文件处理都超时或失败，可能是文件过大或系统资源不足',
                                                     suggestions=[
                                                         '请尝试上传较小的文件',
                                                         '检查系统内存是否充足',
                                                         '如果文件很大，请考虑分批处理',
                                                         '联系管理员检查系统资源'
                                                     ])
                            
                            # 分析处理结果
                            successful_files = [r for r in file_results if 'error' not in r]
                            failed_files = [r for r in file_results if 'error' in r]
                            
                            print("处理完成后的系统状态:")
                            monitor_cpu_usage()
                            monitor_gpu_memory()
                            
                            # 如果有部分文件失败，提供详细信息
                            if failed_files:
                                print(f"部分文件处理失败：{len(failed_files)}个文件")
                                for failed_file in failed_files:
                                    print(f"  - {failed_file.get('filename', '未知文件')}: {failed_file.get('error', '未知错误')}")
                                
                                # 如果失败文件较多，提供建议
                                if len(failed_files) > len(successful_files):
                                    print("大部分文件处理失败，建议检查系统资源或文件格式")
                            
                            # 最终清理GPU内存
                            cleanup_gpu_memory()
                            
                            # 更新系统资源统计
                            system_resources = update_system_resources()
                            print(f"处理完成 - 系统资源：CPU {system_resources['cpu_usage']:.1f}%, 内存 {system_resources['memory_usage']:.1f}%, GPU {system_resources['gpu_memory_usage']:.1f}GB")
                            
                            success_results = [r for r in file_results if r and 'error' not in r]
                            error_results = [r for r in file_results if r and 'error' in r]

                            results = success_results
                            success_set = set(r['filename'] for r in success_results)
                            all_set = set(rel for rel, _p in storage_files)
                            failed_set = all_set - success_set

                            # 准备成功和失败文件列表信息
                            success_files = []
                            failed_files = []
                            
                            # 成功文件列表
                            for name in sorted(success_set):
                                success_files.append({
                                    'filename': name,
                                    'status': 'success'
                                })
                            
                            # 失败文件列表（包含错误信息）
                            error_map = {er['filename']: er.get('error', '未知原因') for er in error_results}
                            for name in sorted(all_set):
                                if name in failed_set:
                                    failed_files.append({
                                        'filename': name,
                                        'error': error_map.get(name, '未知原因'),
                                        'status': 'failed'
                                    })

                            print(f"比对完成：成功 {len(success_set)} 个，失败 {len(failed_set)} 个。")
                            if success_set:
                                print("比对成功文件：")
                                for name in sorted(success_set):
                                    print(f"  ✓ {name}")
                            if error_results:
                                print("比对失败文件及原因：")
                                for name in sorted(all_set):
                                    if name in failed_set:
                                        print(f"  ✗ {name} -> {error_map.get(name, '未知原因')}")
                            
                            if results:
                                results.sort(key=lambda x: x['similarity'], reverse=True)
                                similarities[uploaded_file.filename] = results
                                save_similarities(similarities)
                                print(f"成功处理 {len(results)} 个文件，结果已保存。")
                            else:
                                error = '存储库中无匹配文件'
                            
                            # 准备错误信息和建议
                            error_details = None
                            suggestions = []
                            
                            if failed_files:
                                if len(failed_files) > len(successful_files):
                                    error_details = f"大部分文件处理失败（{len(failed_files)}/{len(failed_files) + len(successful_files)}）"
                                    suggestions = [
                                        '文件可能过大，请尝试上传较小的文件',
                                        '检查系统内存是否充足',
                                        '如果文件很大，请考虑分批处理',
                                        '联系管理员检查系统资源'
                                    ]
                                else:
                                    error_details = f"部分文件处理失败（{len(failed_files)}个）"
                                    suggestions = [
                                        '部分文件处理失败，但主要文件已成功处理',
                                        '可以查看失败文件的详细错误信息',
                                        '如需处理失败文件，请重新上传'
                                    ]
                            
                            _finish_progress(task_id, '完成')
                except Exception as e:
                    error = f"处理文件时出错: {str(e)}"
                    print(f"处理错误: {e}")
                finally:
                    if os.path.exists(upload_path):
                        try:
                            os.remove(upload_path)
                            print(f"已删除上传文件: {upload_path}")
                        except Exception as e:
                            print(f"删除上传文件失败: {upload_path}, 错误: {e}")
                    if temp_file_path and os.path.exists(temp_file_path) and temp_file_path != upload_path:
                        try:
                            os.remove(temp_file_path)
                            print(f"已删除临时文件: {temp_file_path}")
                        except Exception as e:
                            print(f"删除临时文件失败: {temp_file_path}, 错误: {e}")

    if results:
        # 任务完成，清理进程
        if 'task_id' in locals():
            print(f"存储库比较任务 {task_id} 完成")
            
            # 立即强制清理所有相关进程
            print("存储库比较：开始强制清理所有相关进程...")
            try:
                import psutil
                current_pid = os.getpid()
                killed_count = 0
                
                # 获取当前所有进程
                current_processes = set()
                for proc in psutil.process_iter(['pid']):
                    current_processes.add(proc.info['pid'])
                
                # 找出新创建的进程
                if 'initial_processes' in locals():
                    new_processes = current_processes - initial_processes
                    print(f"存储库比较：发现 {len(new_processes)} 个新创建的进程")
                    
                    # 杀死所有新创建的Python进程
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            if proc.info['pid'] in new_processes and proc.info['name'] == 'python' and proc.info['pid'] != current_pid:
                                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                                # 检查是否是multiprocessing相关进程
                                if 'multiprocessing' in cmdline or 'spawn_main' in cmdline or 'storage_compare' in cmdline or 'keyword_search' in cmdline:
                                    print(f"存储库比较：发现相关进程 {proc.info['pid']}，命令行: {cmdline[:100]}...")
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
                    print(f"存储库比较：已终止 {killed_count} 个新创建的进程")
            except Exception as e:
                print(f"存储库比较：强制清理进程时出错: {e}")
            
            # 清理任务状态
            clear_cancelled_task(task_id)
            clear_task_timer(task_id)
            terminate_task_processes(task_id)
            clear_task_processes(task_id)
            # 温和的进程清理：只清理明确属于当前任务的进程
            from utils import cleanup_task_specific_processes
            try:
                cleanup_task_specific_processes(task_id)
            except:
                # 如果温和清理失败，使用传统方法但更加谨慎
                from utils import kill_all_multiprocessing_processes
                kill_all_multiprocessing_processes()
        
        # 清理环境变量
        os.environ.pop('SUPPRESS_INIT_LOGS', None)
        return render_template('storage_compare.html', 
                             results=results, 
                             uploaded_file=uploaded_file, 
                             error=error,
                             error_details=error_details if 'error_details' in locals() else None,
                             suggestions=suggestions if 'suggestions' in locals() else [],
                             success_files=success_files if 'success_files' in locals() else [],
                             failed_files=failed_files if 'failed_files' in locals() else [])
    else:
        # GET请求：用户直接访问或刷新结果页面，重定向到首页
        from flask import redirect, url_for
        print(f"存储库比较GET请求 - 用户: {user_id}，重定向到首页")
        # 清理环境变量
        os.environ.pop('SUPPRESS_INIT_LOGS', None)
        return redirect(url_for('index'))


@storage_compare_bp.route('/system_status', methods=['GET'])
def get_system_status():
    """获取系统状态信息"""
    from utils import get_system_resources, update_system_resources
    
    # 更新系统资源
    system_resources = update_system_resources()
    
    return jsonify({
        'status': 'success',
        'system_resources': system_resources,
        'timestamp': time.time()
    }), 200

@storage_compare_bp.route('/Uploads/temp/<filename>')
def serve_temp_file(filename):
    return send_from_directory(os.path.join('Uploads', 'temp'), filename)
