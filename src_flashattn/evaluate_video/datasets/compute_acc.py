import pandas as pd
import json
from pathlib import Path

def load_as_dataframe(file_path):
    """根据文件扩展名加载为 pandas DataFrame"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    
    elif suffix == '.jsonl':
        # JSONL: one JSON object per line
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return pd.DataFrame(data)
    
    elif suffix == '.json':
        # JSON: assume it's a list of dicts
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of objects.")
        return pd.DataFrame(data)
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Please use .xlsx, .xls, .json, or .jsonl.")

def calculate_accuracy(file_path):
    try:
        # 1. 读取 Excel 文件
        # 如果你的 Excel 文件包含多个 sheet，可以通过 sheet_name 指定
        # df = pd.read_excel(file_path)
        df = load_as_dataframe(file_path)
        
        # 打印列名以确认读取正确
        print("检测到的列名:", df.columns.tolist())

        # 2. 指定关键列名 (根据你的截图)
        gt_col = 'answer'       # 真实标签 (Ground Truth)
        pred_col = 'prediction' # 模型预测结果

        # 检查列是否存在
        if gt_col not in df.columns or pred_col not in df.columns:
            print(f"错误: 未找到列 '{gt_col}' 或 '{pred_col}'，请检查Excel表头。")
            return

        # 3. 数据清洗 (这一步很重要)
        # 确保都是字符串，去除前后空格，并统一转为大写，防止 'A ' != 'A' 的情况
        df[gt_col] = df[gt_col].astype(str).str.strip().str.upper()
        df[pred_col] = df[pred_col].astype(str).str.strip().str.upper()

        # 4. 计算准确率
        # 比较两列是否相等
        matches = df[df[gt_col] == df[pred_col]]
        
        total_samples = len(df)
        correct_count = len(matches)
        
        if total_samples == 0:
            print("数据为空，无法计算。")
            return

        accuracy = correct_count / total_samples

        # 5. 输出结果
        print("-" * 30)
        print(f"📊 评估报告: {file_path}")
        print("-" * 30)
        print(f"总样本数 (Total):    {total_samples}")
        print(f"预测正确数 (Correct):  {correct_count}")
        print(f"准确率 (Accuracy):    {accuracy:.4f} ({accuracy:.2%})")
        print("-" * 30)
        
        # 可选：将错误样本保存出来方便分析
        # wrong_samples = df[df[gt_col] != df[pred_col]]
        # wrong_samples.to_excel("bad_cases.xlsx", index=False)
        # print("错误样本已保存至 bad_cases.xlsx")

    except Exception as e:
        print(f"发生错误: {e}")

# --- 使用示例 ---
# 请将这里的文件名替换为你实际保存的 Excel 文件名
# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/final/Qwen3-VL-8B-Instruct_Video-MME-no-VLLM.xlsx'
# 总样本数 (Total):    2700
# 预测正确数 (Correct):  1947
# 准确率 (Accuracy):    0.7211 (72.11%)

# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_Video-MME-no-VLLM_941.xlsx'
# 总样本数 (Total):    942
# 预测正确数 (Correct):  747
# 准确率 (Accuracy):    0.7930 (79.30%)

# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/output_file.xlsx'
# 总样本数 (Total):    942
# 预测正确数 (Correct):  743
# 准确率 (Accuracy):    0.7887 (78.87%)

# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_Video-MME-no-VLLM_898.xlsx'
# 总样本数 (Total):    899
# 预测正确数 (Correct):  720
# 准确率 (Accuracy):    0.8009 (80.09%)

# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/output_file_898.xlsx'
# 总样本数 (Total):    899
# 预测正确数 (Correct):  718
# 准确率 (Accuracy):    0.7987 (79.87%)
# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/T20260111_G85738573/Qwen3-VL-8B-Instruct_Video-MME.xlsx'
# 总样本数 (Total):    2700
# 预测正确数 (Correct):  1918
# 准确率 (Accuracy):    0.7104 (71.04%)

# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/refine/Qwen3-VL-8B-Instruct/Video-MME/T20260122_G85738573/Qwen3-VL-8B-Instruct_Video-MME.jsonl'
# 总样本数 (Total):    2700
# 预测正确数 (Correct):  1913
# 准确率 (Accuracy):    0.7085 (70.85%)



# MLVU
# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/MLVU/T20260115_G85738573/Qwen3-VL-8B-Instruct_MLVU.xlsx'
# 总样本数 (Total):    793
# 预测正确数 (Correct):  659
# 准确率 (Accuracy):    0.8310 (83.10%)
# 总样本数 (Total):    1438
# 预测正确数 (Correct):  1044
# 准确率 (Accuracy):    0.7260 (72.60%)
# 总样本数 (Total):    2174
# 预测正确数 (Correct):  1635
# 准确率 (Accuracy):    0.7521 (75.21%)

# lvbench
# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/LVBench/T20260115_G85738573/Qwen3-VL-8B-Instruct_LVBench_copy.jsonl'
# acc
# 总样本数 (Total):    1492
# 预测正确数 (Correct):  825
# 准确率 (Accuracy):    0.5529 (55.29%)

# mvbench
# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/MVBench/T20260115_G85738573/Qwen3-VL-8B-Instruct_MVBench.jsonl'
# 总样本数 (Total):    3600
# 预测正确数 (Correct):  2520
# 准确率 (Accuracy):    0.7000 (70.00%)

# videomme fangxin
# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/refine/Qwen3-VL-8B-Instruct/Video-MME/T20260119_G85738573/Qwen3-VL-8B-Instruct_Video-MME.xlsx'
# 总样本数 (Total):    2700
# 预测正确数 (Correct):  1786
# 准确率 (Accuracy):    0.6615 (66.15%)


# lvbench new
# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/refine/Qwen3-VL-8B-Instruct/LVBench/T20260119_G85738573/Qwen3-VL-8B-Instruct_LVBench.jsonl'
# 总样本数 (Total):    1492
# 预测正确数 (Correct):  819
# 准确率 (Accuracy):    0.5489 (54.89%)


# videomme, aks clip offical

# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/sample/method/aks/clip/Qwen3-VL-8B-Instruct/Video-MME/T20260125_G85738573/Qwen3-VL-8B-Instruct_Video-MME.jsonl'
# 总样本数 (Total):    2700
# 预测正确数 (Correct):  1866
# 准确率 (Accuracy):    0.6911 (69.11%)

# videomme, aks blip offical
# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/sample/method/aks/blip/Qwen3-VL-8B-Instruct/Video-MME/T20260126_G85738573/Qwen3-VL-8B-Instruct_Video-MME.jsonl'
# 总样本数 (Total):    2700
# 预测正确数 (Correct):  1838
# 准确率 (Accuracy):    0.6807 (68.07%)


# videomme, embedding easy softmax 64
# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/sample/Qwen3-VL-8B-Instruct/Video-MME/T20260123_G85738573/Qwen3-VL-8B-Instruct_Video-MME.jsonl'
# 总样本数 (Total):    2700
# 预测正确数 (Correct):  1861
# 准确率 (Accuracy):    0.6893 (68.93%)

# file_name = '/data/oceanus_share/shangshouduo-jk/project/results/sample/method/aks/blip/new/Qwen3-VL-8B-Instruct/Video-MME/T20260127_G85738573/Qwen3-VL-8B-Instruct_Video-MME.jsonl'

# file_name = '/data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/q-frame/longclip/Qwen3-VL-8B-Instruct/Video-MME/T20260203_G85738573/Qwen3-VL-8B-Instruct_Video-MME.jsonl'
# 总样本数 (Total):    2700
# 预测正确数 (Correct):  1681
# 准确率 (Accuracy):    0.6226 (62.26%)

# aks videomme clip
# file_name = '/data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/aks/clip/Qwen3-VL-8B-Instruct/Video-MME/T20260203_G85738573/Qwen3-VL-8B-Instruct_Video-MME.jsonl'
# 总样本数 (Total):    2700
# 预测正确数 (Correct):  1866
# 准确率 (Accuracy):    0.6911 (69.11%)

# backbone lvbench
# file_name = '/data/oceanus_share/shangshouduo-jk/myproject/output/results/backbone/Qwen3-VL-8B-Instruct/LVBench/T20260204_G85738573/Qwen3-VL-8B-Instruct_LVBench_copy.jsonl'
# 总样本数 (Total):    1549
# 预测正确数 (Correct):  858
# 准确率 (Accuracy):    0.5539 (55.39%)

# 总样本数 (Total):    1492
# 预测正确数 (Correct):  825
# 准确率 (Accuracy):    0.5529 (55.29%)

# q-frame lvbench
# file_name = '/data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/q-frame/longclip/Qwen3-VL-8B-Instruct/LVBench/T20260205_G85738573/Qwen3-VL-8B-Instruct_LVBench.jsonl'
# 总样本数 (Total):    1549
# 预测正确数 (Correct):  600
# 准确率 (Accuracy):    0.3873 (38.73%)

# q-frame mlvu
# file_name = '/data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/q-frame/longclip/Qwen3-VL-8B-Instruct/MLVU/T20260205_G85738573/Qwen3-VL-8B-Instruct_MLVU.jsonl'
# 总样本数 (Total):    2174
# 预测正确数 (Correct):  1429
# 准确率 (Accuracy):    0.6573 (65.73%)

# q-frame videomme
# file_name = "/data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/q-frame/longclip/Qwen3-VL-8B-Instruct/Video-MME/T20260203_G85738573/Qwen3-VL-8B-Instruct_Video-MME.jsonl"
# 总样本数 (Total):    2700
# 预测正确数 (Correct):  1681
# 准确率 (Accuracy):    0.6226 (62.26%)

# q-frame mvbench
# file_name = "/data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/q-frame/longclip/Qwen3-VL-8B-Instruct/MVBench/T20260205_G85738573/Qwen3-VL-8B-Instruct_MVBench.jsonl"
# 总样本数 (Total):    4000
# 预测正确数 (Correct):  2460
# 准确率 (Accuracy):    0.6150 (61.50%)

# baseline mmvbench
# file_name = "/data/oceanus_share/shangshouduo-jk/myproject/output/results/backbone/Qwen3-VL-8B-Instruct/MVBench/T20260206_G85738573/Qwen3-VL-8B-Instruct_MVBench.xlsx"
# 总样本数 (Total):    4000
# 预测正确数 (Correct):  2754
# 准确率 (Accuracy):    0.6885 (68.85%)

# aks mvbench
file_name = "/data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/aks/clip/Qwen3-VL-8B-Instruct/MVBench/T20260205_G85738573/Qwen3-VL-8B-Instruct_MVBench.jsonl"
# 总样本数 (Total):    4000
# 预测正确数 (Correct):  2610
# 准确率 (Accuracy):    0.6525 (65.25%)

# file_name = "/data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/aks/clip/Qwen3-VL-8B-Instruct/MLVU/T20260206_G85738573/Qwen3-VL-8B-Instruct_MLVU.jsonl"
 

calculate_accuracy(file_name)