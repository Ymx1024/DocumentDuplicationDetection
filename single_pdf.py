from typing import Tuple, Dict, Any
from utils import (
    extract_formatted_text,
    compute_similarity,
    compute_differences,
    is_task_cancelled,
)


def compare_pdf_documents(
    task_id: str,
    orig_file_info: Tuple[str, str],
    comp_file_info: Tuple[str, str],
) -> Dict[str, Any]:
    """
    单文件_pdf 比对实现（.pdf）。
    使用通用行提取，保持与结果页结构一致。
    """
    orig_path, orig_filename = orig_file_info
    comp_path, comp_filename = comp_file_info

    if is_task_cancelled(task_id):
        raise Exception("任务在开始前被取消")

    orig_lines, _, _ = extract_formatted_text(orig_path, task_id)
    if is_task_cancelled(task_id):
        raise Exception("任务在提取原文后被取消")

    comp_lines, _, _ = extract_formatted_text(comp_path, task_id)
    if is_task_cancelled(task_id):
        raise Exception("任务在提取对比文件后被取消")

    if not orig_lines or not comp_lines:
        raise Exception("无法从一个或两个文件中提取文本")

    orig_text = "\n".join(orig_lines)
    comp_text = "\n".join(comp_lines)

    similarity = compute_similarity(orig_text, comp_text)
    diffs = compute_differences(orig_text, comp_text)

    enhanced_diffs = []
    for diff in diffs:
        enhanced_diffs.append({
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
        })

    display_similarity = round(similarity * 100, 2)
    if enhanced_diffs and display_similarity >= 100.0:
        display_similarity = 99.99

    results = {
        'similarity': display_similarity,
        'differences': enhanced_diffs,
        'original_lines': orig_lines,
        'comparison_lines': comp_lines,
        'original_filename': orig_filename,
        'comparison_filename': comp_filename,
        'total_lines_orig': len(orig_lines),
        'total_lines_comp': len(comp_lines),
        'diff_count': len(enhanced_diffs),
        'has_differences': len(enhanced_diffs) > 0,
    }

    return results


