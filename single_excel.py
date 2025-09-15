from typing import Tuple, Dict, Any, List
from utils import (
    extract_formatted_text,
    compute_similarity,
    compute_differences,
    is_task_cancelled,
)
import re


def compare_excel_documents(
    task_id: str,
    orig_file_info: Tuple[str, str],
    comp_file_info: Tuple[str, str],
) -> Dict[str, Any]:
    """
    单文件_excel 比对实现（.xls / .xlsx）。
    与 Word 专用模块保持一致的返回结构，前端复用同一结果页面。
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

    def normalize_excel_lines(lines: List[str]) -> List[str]:
        token_regex = re.compile(r"\d+\.\d+|\d+|[A-Za-z]+|[\u4e00-\u9fff]+")
        normalized: List[str] = []
        for line in lines:
            tokens = token_regex.findall(line or "")
            normalized.append(" ".join(tokens).strip())
        return normalized

    norm_orig_lines = normalize_excel_lines(orig_lines)
    norm_comp_lines = normalize_excel_lines(comp_lines)
    orig_text = "\n".join(norm_orig_lines)
    comp_text = "\n".join(norm_comp_lines)

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


