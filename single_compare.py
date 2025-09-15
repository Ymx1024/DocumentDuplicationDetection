from flask import Blueprint, request, render_template, send_from_directory, jsonify, url_for
from utils import allowed_file, compute_similarity, get_common_words, compute_differences, extract_formatted_text, mark_task_cancelled, is_task_cancelled, clear_cancelled_task, start_task_timer, clear_task_timer, terminate_task_processes, clear_task_processes
from single_word import compare_word_documents
from single_excel import compare_excel_documents
from single_pdf import compare_pdf_documents
import os
import uuid
import threading
import time

single_compare_bp = Blueprint('single_compare', __name__)

# 用于存储异步任务的结果
TASK_RESULTS = {}

def run_comparison_task(task_id, orig_file_info, comp_file_info):
    """在后台线程中运行文件比较任务"""
    orig_path, orig_filename = orig_file_info
    comp_path, comp_filename = comp_file_info
    
    try:
        if is_task_cancelled(task_id):
            raise Exception("任务在开始前被取消")

        # 若为 Word / Excel 文档，直接走专用实现
        ext = os.path.splitext(orig_filename)[1].lower()
        if ext in ['.doc', '.docx']:
            results = compare_word_documents(
                task_id,
                (orig_path, orig_filename),
                (comp_path, comp_filename)
            )
            TASK_RESULTS[task_id] = {'status': 'completed', 'result': results}
            print(f"✅ 单文件比较任务 {task_id} 完成 (Word 专用)")
            return
        if ext in ['.xls', '.xlsx']:
            results = compare_excel_documents(
                task_id,
                (orig_path, orig_filename),
                (comp_path, comp_filename)
            )
            TASK_RESULTS[task_id] = {'status': 'completed', 'result': results}
            print(f"✅ 单文件比较任务 {task_id} 完成 (Excel 专用)")
            return

        orig_lines, _, _ = extract_formatted_text(orig_path, task_id)
        if is_task_cancelled(task_id):
            raise Exception("任务在提取原文后被取消")

        comp_lines, _, _ = extract_formatted_text(comp_path, task_id)
        if is_task_cancelled(task_id):
            raise Exception("任务在提取对比文件后被取消")

        if not orig_lines or not comp_lines:
            raise Exception("无法从一个或两个文件中提取文本")

        orig_text = '\\n'.join(orig_lines)
        comp_text = '\\n'.join(comp_lines)
        
        similarity = compute_similarity(orig_text, comp_text)
        differences = compute_differences(orig_text, comp_text)

        # 为前端优化差异数据结构
        enhanced_differences = []
        for i, diff in enumerate(differences):
            enhanced_diff = {
                'id': diff['id'],
                'type': diff['type'],
                'original': diff['original'],
                'new': diff['new'],
                'start_idx_orig': diff['start_idx_orig'],
                'end_idx_orig': diff['end_idx_orig'],
                'start_idx_new': diff['start_idx_new'],
                'end_idx_new': diff['end_idx_new'],
                'line_count_orig': diff['end_idx_orig'] - diff['start_idx_orig'],
                'line_count_new': diff['end_idx_new'] - diff['start_idx_new'],
                'preview_orig': diff['original'][:100] + '...' if len(diff['original']) > 100 else diff['original'],
                'preview_new': diff['new'][:100] + '...' if len(diff['new']) > 100 else diff['new']
            }
            enhanced_differences.append(enhanced_diff)

        results = {
            'similarity': round(similarity * 100, 2),
            'differences': enhanced_differences,
            'original_lines': orig_lines,
            'comparison_lines': comp_lines,
            'original_filename': orig_filename,
            'comparison_filename': comp_filename,
            'total_lines_orig': len(orig_lines),
            'total_lines_comp': len(comp_lines),
            'diff_count': len(enhanced_differences),
            'has_differences': len(enhanced_differences) > 0
        }
        
        TASK_RESULTS[task_id] = {'status': 'completed', 'result': results}
        print(f"✅ 单文件比较任务 {task_id} 完成")

    except Exception as e:
        error_message = str(e)
        if "任务" in error_message: # 如果是手动取消的，就用取消信息
             TASK_RESULTS[task_id] = {'status': 'cancelled', 'error': error_message}
             print(f"🛑 单文件比较任务 {task_id} 已被取消: {error_message}")
        else:
             TASK_RESULTS[task_id] = {'status': 'error', 'error': error_message}
             print(f"❌ 单文件比较任务 {task_id} 失败: {error_message}")
    finally:
        # 清理临时文件
        for p in [orig_path, comp_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    print(f"删除临时文件失败: {p}, 错误: {e}")
        clear_cancelled_task(task_id)


@single_compare_bp.route('/progress/<task_id>')
def progress_page(task_id):
    """显示任务进度的等待页面"""
    return render_template('progress.html', task_id=task_id, task_type='single_compare')

@single_compare_bp.route('/status/<task_id>')
def task_status(task_id):
    """获取任务状态 (用于JS轮询)"""
    task = TASK_RESULTS.get(task_id)
    if not task:
        return jsonify({'status': 'pending'})
    
    if task['status'] == 'completed':
        return jsonify({
            'status': 'completed',
            'result_url': url_for('single_compare.result_page', task_id=task_id)
        })
    elif task['status'] in ['error', 'cancelled']:
         return jsonify({'status': 'error', 'message': task.get('error', '未知错误')})

    return jsonify({'status': 'pending'})

@single_compare_bp.route('/result/<task_id>')
def result_page(task_id):
    """显示任务结果页面"""
    task = TASK_RESULTS.pop(task_id, None) # 获取结果后从内存中移除
    if not task or task['status'] != 'completed':
        error_msg = "任务不存在或未完成"
        if task and 'error' in task:
            error_msg = task['error']
        return render_template('index.html', error=error_msg)
    
    results = task.get('result')
    filename = (results or {}).get('original_filename', '')
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.doc', '.docx']:
        return render_template('single_compare_word.html', results=results)
    if ext in ['.xls', '.xlsx']:
        return render_template('single_compare_excel.html', results=results)
    if ext in ['.pdf']:
        return render_template('single_compare_pdf.html', results=results)
    return render_template('single_compare_excel.html', results=results)


@single_compare_bp.route('/cancel/<task_id>', methods=['POST'])
def cancel_single_compare_task(task_id):
    """取消单文件比较任务"""
    print(f"单文件比较任务 {task_id} 取消请求")
    mark_task_cancelled(task_id)
    # 确保任务状态被记录
    if task_id not in TASK_RESULTS:
        TASK_RESULTS[task_id] = {'status': 'cancelled', 'error': '任务已被用户取消'}
    return jsonify({'status': 'cancelled', 'message': f'任务 {task_id} 已取消'})


@single_compare_bp.route('/diff/<task_id>/<diff_id>')
def get_diff_details(task_id, diff_id):
    """获取特定差异的详细信息"""
    task = TASK_RESULTS.get(task_id)
    if not task or task['status'] != 'completed':
        return jsonify({'error': '任务不存在或未完成'}), 404
    
    result = task.get('result')
    if not result:
        return jsonify({'error': '结果数据不存在'}), 404
    
    # 查找指定的差异
    target_diff = None
    for diff in result['differences']:
        if diff['id'] == diff_id:
            target_diff = diff
            break
    
    if not target_diff:
        return jsonify({'error': '差异不存在'}), 404
    
    # 返回详细的差异信息
    return jsonify({
        'diff': target_diff,
        'context': {
            'original_filename': result['original_filename'],
            'comparison_filename': result['comparison_filename'],
            'similarity': result['similarity']
        }
    })


@single_compare_bp.route('/export/<task_id>')
def export_comparison_result(task_id):
    """导出比较结果为JSON格式"""
    task = TASK_RESULTS.get(task_id)
    if not task or task['status'] != 'completed':
        return jsonify({'error': '任务不存在或未完成'}), 404
    
    result = task.get('result')
    if not result:
        return jsonify({'error': '结果数据不存在'}), 404
    
    # 创建导出数据
    export_data = {
        'metadata': {
            'export_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'original_filename': result['original_filename'],
            'comparison_filename': result['comparison_filename'],
            'similarity': result['similarity'],
            'total_differences': result['diff_count'],
            'total_lines_original': result['total_lines_orig'],
            'total_lines_comparison': result['total_lines_comp']
        },
        'differences': result['differences'],
        'summary': {
            'modified_count': len([d for d in result['differences'] if d['type'] == 'modified']),
            'added_count': len([d for d in result['differences'] if d['type'] == 'added']),
            'deleted_count': len([d for d in result['differences'] if d['type'] == 'deleted'])
        }
    }
    
    return jsonify(export_data)


@single_compare_bp.route('/filter/<task_id>')
def filter_differences(task_id):
    """根据条件过滤差异"""
    task = TASK_RESULTS.get(task_id)
    if not task or task['status'] != 'completed':
        return jsonify({'error': '任务不存在或未完成'}), 404
    
    result = task.get('result')
    if not result:
        return jsonify({'error': '结果数据不存在'}), 404
    
    # 获取过滤参数
    diff_type = request.args.get('type')  # modified, added, deleted
    search_text = request.args.get('search', '').strip()
    min_length = request.args.get('min_length', type=int)
    max_length = request.args.get('max_length', type=int)
    
    # 过滤差异
    filtered_diffs = result['differences']
    
    # 按类型过滤
    if diff_type and diff_type in ['modified', 'added', 'deleted']:
        filtered_diffs = [d for d in filtered_diffs if d['type'] == diff_type]
    
    # 按文本内容搜索
    if search_text:
        search_lower = search_text.lower()
        filtered_diffs = [d for d in filtered_diffs 
                         if search_lower in d['original'].lower() or 
                            search_lower in d['new'].lower()]
    
    # 按长度过滤
    if min_length is not None:
        filtered_diffs = [d for d in filtered_diffs 
                         if len(d['original']) >= min_length or len(d['new']) >= min_length]
    
    if max_length is not None:
        filtered_diffs = [d for d in filtered_diffs 
                         if len(d['original']) <= max_length and len(d['new']) <= max_length]
    
    return jsonify({
        'filtered_differences': filtered_diffs,
        'total_count': len(filtered_diffs),
        'original_count': len(result['differences']),
        'filters_applied': {
            'type': diff_type,
            'search': search_text,
            'min_length': min_length,
            'max_length': max_length
        }
    })


@single_compare_bp.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'files' not in request.files:
            return jsonify({'error': '请上传两个文件进行比对'}), 400
        
        files = request.files.getlist('files')
        if len(files) != 2:
            return jsonify({'error': '请正好选择两个文件'}), 400

        orig_file, comp_file = files
        if orig_file.filename == '' or comp_file.filename == '':
            return jsonify({'error': '请确保两个文件都已选择'}), 400
        if not (allowed_file(orig_file.filename) and allowed_file(comp_file.filename)):
            return jsonify({'error': '不支持的文件类型'}), 400

        ext1 = orig_file.filename.rsplit('.', 1)[1].lower()
        ext2 = comp_file.filename.rsplit('.', 1)[1].lower()
        if ext1 != ext2:
            return jsonify({'error': '两个文件必须为相同格式'}), 400
            
        task_id = str(uuid.uuid4())
        
        # 保存文件（命名模式：比对模式_文件类型_角色_短唯一标识.ext）
        def classify_file_type(ext: str) -> str:
            if ext in ['doc', 'docx']:
                return 'word'
            if ext in ['xls', 'xlsx']:
                return 'excel'
            if ext in ['pdf']:
                return 'pdf'
            if ext in ['txt']:
                return 'text'
            if ext in ['yml', 'yaml']:
                return 'yaml'
            return 'other'

        mode_prefix = 'single'
        file_type = classify_file_type(ext1)
        short_id = uuid.uuid4().hex[:8]
        orig_filename_saved = f"{mode_prefix}_{file_type}_orig_{short_id}.{ext1}"
        comp_filename_saved = f"{mode_prefix}_{file_type}_comp_{short_id}.{ext2}"
        orig_path = os.path.join('Uploads', orig_filename_saved)
        comp_path = os.path.join('Uploads', comp_filename_saved)
        
        try:
            orig_file.save(orig_path)
            comp_file.save(comp_path)
        except Exception as e:
            return jsonify({'error': f'保存文件失败: {e}'}), 500

        # 标记任务开始
        TASK_RESULTS[task_id] = {'status': 'pending'}
        
        # 启动后台线程执行比对
        thread = threading.Thread(target=run_comparison_task, args=(
            task_id,
            (orig_path, orig_file.filename),
            (comp_path, comp_file.filename)
        ))
        thread.start()
        
        print(f"🚀 单文件比较任务 {task_id} 已在后台启动")

        # 立即返回，让前端跳转到进度页
        return jsonify({
            'status': 'pending',
            'task_id': task_id,
            'progress_url': url_for('single_compare.progress_page', task_id=task_id)
        })

    # GET请求重定向到首页
    from flask import redirect
    return redirect(url_for('index'))


