from typing import Tuple, Dict, Any, List
from utils import (
    extract_formatted_text,
    compute_similarity,
    compute_differences,
    is_task_cancelled,
)


def compare_word_documents(
    task_id: str,
    orig_file_info: Tuple[str, str],
    comp_file_info: Tuple[str, str],
) -> Dict[str, Any]:
    """
    单文件_word 比对实现（.doc / .docx）。
    返回结构与结果页保持一致。
    """
    orig_path, orig_filename = orig_file_info
    comp_path, comp_filename = comp_file_info

    # 1) 文本提取
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

    orig_text = '\n'.join(orig_lines)
    comp_text = '\n'.join(comp_lines)

    # 使用与差异算法一致的“真实行”分割，确保行索引精准对齐
    import re
    def split_into_real_lines(text: str) -> List[str]:
        normalized_text = text.replace('\r\n', '\n').replace('\n', '\n')
        initial_lines = normalized_text.split('\n')
        real_lines: List[str] = []
        for line in initial_lines:
            curr = line.strip()
            if not curr:
                continue
            if len(curr) > 100:
                sentence_parts = re.split(r'(?<=[。！？；;:,，、])\s*', curr)
                for sent in sentence_parts:
                    s = (sent or '').strip()
                    if not s:
                        continue
                    if len(s) > 100:
                        phrase_parts = re.split(r'[、，,：:；;]|\s{2,}|\t+', s)
                        for ph in phrase_parts:
                            p = (ph or '').strip()
                            if not p:
                                continue
                            if len(p) > 100:
                                number_parts = re.split(r'(?:\d+\.?\d*\s*){1,}', p)
                                for np in number_parts:
                                    nps = (np or '').strip()
                                    if nps:
                                        real_lines.append(nps)
                            else:
                                real_lines.append(p)
                    else:
                        real_lines.append(s)
            else:
                real_lines.append(curr)
        return real_lines

    display_orig_lines = split_into_real_lines(orig_text)
    display_comp_lines = split_into_real_lines(comp_text)

    # 2) 相似度与差异
    similarity = compute_similarity(orig_text, comp_text)
    diffs = compute_differences(orig_text, comp_text)

    # 3) 前端友好结构
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

    # 显示相似度保护：有差异则不显示 100.00%
    display_similarity = round(similarity * 100, 2)
    if enhanced_diffs and display_similarity >= 100.0:
        display_similarity = 99.99

    results = {
        'similarity': display_similarity,
        'differences': enhanced_diffs,
        'original_lines': display_orig_lines,
        'comparison_lines': display_comp_lines,
        'original_filename': orig_filename,
        'comparison_filename': comp_filename,
        'total_lines_orig': len(display_orig_lines),
        'total_lines_comp': len(display_comp_lines),
        'diff_count': len(enhanced_diffs),
        'has_differences': len(enhanced_diffs) > 0,
    }

    return results


