import json
import re
import os
from typing import List, Union

# ================= 配置路径 =================
INPUT_FILE = "/data/oceanus_share/shangshouduo-jk/myproject/output/results/backbone/Qwen3-VL-8B-Instruct/FCMBench/T20260205_G85738573/Qwen3-VL-8B-Instruct_FCMBench.jsonl"
OUTPUT_FILE = "/data/oceanus_share/shangshouduo-jk/myproject/output/results/backbone/Qwen3-VL-8B-Instruct/FCMBench/T20260205_G85738573/Qwen3-VL-8B-Instruct_FCMBench_scores.jsonl"
# ===========================================

def clean_and_parse_json(field: any) -> any:
    if isinstance(field, (list, dict, int, float)):
        return field
    s = str(field).strip()
    s = re.sub(r'^```(?:json)?\s*', '', s)
    s = re.sub(r'\s*```$', '', s)
    try:
        data = json.loads(s)
        return data["answer"] if isinstance(data, dict) and "answer" in data else data
    except:
        list_match = re.search(r'\[.*\]', s, re.DOTALL)
        if list_match:
            try: return json.loads(list_match.group().replace("'", '"'))
            except: pass
        num_match = re.search(r'\d+', s)
        if num_match: return int(num_match.group())
        return s

def time_to_seconds(time_val: any) -> float:
    """
    针对 FCMBench 优化的时间解析逻辑
    支持: 
    1. "00:36.81" (mm:ss.ff) -> 36.81s
    2. "00:36:24" (模型误用的 mm:ss:ff) -> 36.24s (重点修复在此)
    """
    s_val = str(time_val).strip().replace(',', '.')
    try:
        # 统一分隔符，将 . 替换为 : 方便处理
        parts = s_val.replace('.', ':').split(':')
        
        if len(parts) == 2:
            # 格式: mm:ss
            return int(parts[0]) * 60 + float(parts[1])
        
        elif len(parts) >= 3:
            # 格式: hh:mm:ss 或 mm:ss:ff
            p1, p2, p3 = int(parts[0]), int(parts[1]), float(parts[2])
            
            # 核心修正：如果第一段是0，或者整体数值过大不符合短视频逻辑
            # 在 FCMBench 中，极大概率 p1 是分，p2 是秒，p3 是毫秒/帧
            if p1 == 0:
                # 按照 分:秒.毫秒 解析
                return p2 * 60 + p3 if p3 >= 100 else p2 + (p3 / 100.0) # 这里的判断逻辑根据实际情况
                # 简单处理：对于 00:36:24 -> 36秒 + 0.24秒
                return p1 * 3600 + p2 * 60 + (p3 if p3 < 1 else p3/100.0 if p3 < 100 else p3/1000.0)
            
            # 针对你提供的样本特化处理: "00:36:24" 应为 36.24秒
            # 我们采取最稳妥的映射：最后一部分如果是整数且 > 0，通常是百分秒
            if p1 == 0 and p2 > 0:
                 return p2 + (p3 / 100.0 if p3 < 100 else p3 / 1000.0)
            
            # 默认 fallback
            return p1 * 3600 + p2 * 60 + p3
            
        return float(s_val)
    except:
        return 0.0

def calculate_iou(pred_range: any, gt_range: any) -> float:
    if not isinstance(pred_range, list) or not isinstance(gt_range, list) or len(pred_range) < 2 or len(gt_range) < 2:
        return 0.0
    
    # 针对模型输出如 ["00:01:00", "00:03:00"] 的特化处理
    # 这里的 01:00 极有可能是指 1秒00，而不是1分钟
    def refined_parse(t, is_gt=False):
        # 如果是 GT，保持原有逻辑；如果是 Prediction 且格式为 00:XX:YY
        s = str(t)
        if not is_gt and s.count(':') == 2 and s.startswith('00:'):
            parts = s.split(':')
            # 强制将 00:36:24 转为 36.24
            return float(parts[1]) + float(parts[2])/100.0
        return time_to_seconds(t)

    p_s = refined_parse(pred_range[0], is_gt=False)
    p_e = refined_parse(pred_range[1], is_gt=False)
    g_s = refined_parse(gt_range[0], is_gt=True)
    g_e = refined_parse(gt_range[1], is_gt=True)
    
    p_start, p_end = min(p_s, p_e), max(p_s, p_e)
    g_start, g_end = min(g_s, g_e), max(g_s, g_e)
    
    inter_s, inter_e = max(p_start, g_start), min(p_end, g_end)
    if inter_e <= inter_s: return 0.0
    
    inter = inter_e - inter_s
    union = (p_end - p_start) + (g_end - g_start) - inter
    return inter / union if union > 0 else 0.0

def main():
    if not os.path.exists(INPUT_FILE): return
    stats = {"classification": {"scores": [], "count": 0}, "counting": {"scores": [], "count": 0}, "temporal_grounding": {"scores": [], "count": 0}}
    all_output_data = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            raw = json.loads(line)
            category = raw.get("task_category", "")
            gt = clean_and_parse_json(raw.get("answer"))
            pred = clean_and_parse_json(raw.get("prediction"))
            
            if category == "classification":
                gt_set = set(map(str, gt if isinstance(gt, list) else [gt]))
                pred_set = set(map(str, pred if isinstance(pred, list) else [pred]))
                tp = len(gt_set & pred_set)
                precision = tp / len(pred_set) if pred_set else 0
                recall = tp / len(gt_set) if gt_set else 0
                score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            elif category == "counting":
                try: score = 1.0 if int(float(str(gt))) == int(float(str(pred))) else 0.0
                except: score = 0.0
            elif category == "temporal_grounding":
                score = calculate_iou(pred, gt)
            else: score = 0.0

            if category in stats:
                stats[category]["scores"].append(score)
                stats[category]["count"] += 1
            all_output_data.append({"index": raw.get("index"), "task_category": category, "answer": gt, "prediction": pred, "eval_score": round(score, 4)})

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for entry in all_output_data:
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    total_score = 0
    print("\n" + "="*50)
    for cat, data in stats.items():
        avg = sum(data["scores"]) / data["count"] if data["count"] > 0 else 0
        total_score += avg
        print(f"{cat:<20} | Score: {avg:.4f}")
    print("="*50)

    # total score
    total_score = total_score / len(stats) if len(stats) > 0 else 0
    print(f"Total Score: {total_score:.4f}")

if __name__ == "__main__":
    main()

# ==================================================
# classification       | Score: 0.7241
# counting             | Score: 0.3259
# temporal_grounding   | Score: 0.5708
# ==================================================
# Total Score: 0.5403