# import json
# from collections import defaultdict

# # ===== 配置 =====
# JSONL_PATH = "/data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/MLVU/T20260115_G85738573/Qwen3-VL-8B-Instruct_MLVU.jsonl"

# # 用于按 question_type 统计
# type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

# # 读取 jsonl 文件
# with open(JSONL_PATH, 'r', encoding='utf-8') as f:
#     for line in f:
#         if not line.strip():
#             continue
#         item = json.loads(line)
#         qtype = item["question_type"]
#         answer = item["answer"]
#         pred = item["prediction"]

#         type_stats[qtype]["total"] += 1
#         if pred == answer:
#             type_stats[qtype]["correct"] += 1

# # 计算每个类型的准确率
# type_accuracies = {}
# print("📊 Per-question-type Accuracy:")
# print("-" * 50)
# for qtype in sorted(type_stats.keys()):
#     stats = type_stats[qtype]
#     acc = stats["correct"] / stats["total"]
#     type_accuracies[qtype] = acc
#     print(f"{qtype:<20} | Acc: {acc:.4f} ({stats['correct']}/{stats['total']})")

# # 计算 MLVU-Avg
# mlvu_avg = sum(type_accuracies.values()) / len(type_accuracies)
# print("-" * 50)
# print(f"🎯 MLVU-Avg (macro average over {len(type_accuracies)} types): {mlvu_avg:.4f}")
# print(f"📈 Overall Accuracy (for reference): {sum(s['correct'] for s in type_stats.values()) / sum(s['total'] for s in type_stats.values()):.4f}")



import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

def load_as_dataframe(file_path):
    """根据文件扩展名加载为 pandas DataFrame"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    
    elif suffix == '.jsonl':
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return pd.DataFrame(data)
    
    elif suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must be a list of objects.")
        return pd.DataFrame(data)
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Please use .xlsx, .xls, .json, or .jsonl.")

def evaluate_mlvu_avg(file_path):
    try:
        # 1. 加载数据
        df = load_as_dataframe(file_path)
        print(f"✅ Loaded {len(df)} samples from: {file_path}")
        print("Detected columns:", df.columns.tolist())

        # 2. 必需字段
        required_cols = ["question_type", "answer", "prediction"]
        # required_cols = ["task", "answer", "prediction"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing required column: '{col}'")

        # 3. 数据清洗：统一为字符串、去空格、转大写（仅对 answer/prediction）
        df["answer"] = df["answer"].astype(str).str.strip().str.upper()
        df["prediction"] = df["prediction"].astype(str).str.strip().str.upper()
        # question_type 保持原样（通常为小写字符串如 'count'）

        # 4. 按 question_type 统计
        type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for _, row in df.iterrows():
            qtype = row["question_type"]
            # qtype = row["task"]
            ans = row["answer"]
            pred = row["prediction"]

            type_stats[qtype]["total"] += 1
            if pred == ans:
                type_stats[qtype]["correct"] += 1

        # 5. 计算每类准确率
        type_accuracies = {}
        print("\n📊 Per-question-type Accuracy:")
        print("-" * 50)
        for qtype in sorted(type_stats.keys()):
            stats = type_stats[qtype]
            acc = stats["correct"] / stats["total"]
            type_accuracies[qtype] = acc
            print(f"{qtype:<20} | Acc: {acc:.4f} ({stats['correct']}/{stats['total']})")

        # 6. 计算 MLVU-Avg (macro average)
        mlvu_avg = sum(type_accuracies.values()) / len(type_accuracies)
        overall_acc = sum(s["correct"] for s in type_stats.values()) / sum(s["total"] for s in type_stats.values())

        print("-" * 50)
        print(f"🎯 MLVU-Avg (macro average over {len(type_accuracies)} types): {mlvu_avg:.4f}")
        print(f"📈 Overall Accuracy (micro average): {overall_acc:.4f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

# ==============================
# 使用示例（请修改为你自己的路径）
# ==============================

if __name__ == "__main__":
    # 支持任意格式：.jsonl / .json / .xlsx
    # FILE_PATH = "/data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/MLVU/T20260115_G85738573/Qwen3-VL-8B-Instruct_MLVU.xlsx"
    # FILE_PATH = "/data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/MVBench/T20260115_G85738573/Qwen3-VL-8B-Instruct_MVBench.jsonl"
    # FILE_PATH = "/data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/LVBench/T20260115_G85738573/Qwen3-VL-8B-Instruct_LVBench_copy.jsonl"
    # FILE_PATH = "/data/oceanus_share/shangshouduo-jk/project/results/refine/Qwen3-VL-8B-Instruct/MLVU/T20260121_G85738573/Qwen3-VL-8B-Instruct_MLVU.jsonl"
    # FILE_PATH = "/path/to/results.xlsx"
    # FILE_PATH = "/path/to/results.json"
    # FILE_PATH = "/data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/aks/clip/Qwen3-VL-8B-Instruct/MLVU/T20260205_G85738573/Qwen3-VL-8B-Instruct_MLVU.jsonl"
    FILE_PATH = "/data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/aks/clip/Qwen3-VL-8B-Instruct/MLVU/T20260206_G85738573/Qwen3-VL-8B-Instruct_MLVU.jsonl"
    evaluate_mlvu_avg(FILE_PATH)

# mlvu数据集
# 📊 Per-question-type Accuracy:
# --------------------------------------------------
# anomaly_reco         | Acc: 0.7800 (156/200)
# count                | Acc: 0.2039 (42/206)
# ego                  | Acc: 0.7273 (256/352)
# findNeedle           | Acc: 0.8169 (290/355)
# order                | Acc: 0.7490 (194/259)
# plotQA               | Acc: 0.8479 (457/539)
# topic_reasoning      | Acc: 0.9125 (240/263)
# --------------------------------------------------
# 🎯 MLVU-Avg (macro average over 7 types): 0.7196
# 📈 Overall Accuracy (micro average): 0.7521


# 📊 Per-question-type Accuracy:
# --------------------------------------------------
# anomaly_reco         | Acc: 0.7550 (151/200)
# count                | Acc: 0.2524 (52/206)
# ego                  | Acc: 0.6960 (245/352)
# findNeedle           | Acc: 0.8085 (287/355)
# order                | Acc: 0.6641 (172/259)
# plotQA               | Acc: 0.8404 (453/539)
# topic_reasoning      | Acc: 0.9049 (238/263)
# --------------------------------------------------
# 🎯 MLVU-Avg (macro average over 7 types): 0.7031
# 📈 Overall Accuracy (micro average): 0.7351


# mvbench数据集
# 📊 Per-question-type Accuracy:
# --------------------------------------------------
# action_antonym       | Acc: 0.8050 (161/200)
# action_count         | Acc: 0.5700 (114/200)
# action_localization  | Acc: 0.3900 (78/200)
# action_prediction    | Acc: 0.8200 (164/200)
# action_sequence      | Acc: 0.7900 (158/200)
# character_order      | Acc: 0.8200 (164/200)
# counterfactual_inference | Acc: 0.6600 (132/200)
# egocentric_navigation | Acc: 0.3650 (73/200)
# fine_grained_action  | Acc: 0.4800 (96/200)
# moving_attribute     | Acc: 0.9250 (185/200)
# moving_count         | Acc: 0.7000 (140/200)
# moving_direction     | Acc: 0.6450 (129/200)
# object_existence     | Acc: 0.8700 (174/200)
# object_interaction   | Acc: 0.8000 (160/200)
# object_shuffle       | Acc: 0.4200 (84/200)
# scene_transition     | Acc: 0.9300 (186/200)
# state_change         | Acc: 0.7600 (152/200)
# unexpected_action    | Acc: 0.8500 (170/200)
# --------------------------------------------------
# 🎯 MLVU-Avg (macro average over 18 types): 0.7000
# 📈 Overall Accuracy (micro average): 0.7000

# anomaly_reco         | Acc: 0.7150 (143/200)
# count                | Acc: 0.2573 (53/206)
# ego                  | Acc: 0.6449 (227/352)
# findNeedle           | Acc: 0.8085 (287/355)
# order                | Acc: 0.5637 (146/259)
# plotQA               | Acc: 0.7755 (418/539)
# topic_reasoning      | Acc: 0.9125 (240/263)
# --------------------------------------------------
# 🎯 MLVU-Avg (macro average over 7 types): 0.6682
# 📈 Overall Accuracy (micro average): 0.6964