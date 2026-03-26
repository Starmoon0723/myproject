import pandas as pd

# 定义固定前缀
prefix = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option. Question: "

input_file = "/data/oceanus_share/shangshouduo-jk/project/datasets/Video-MME/Video-MME_copy.tsv"
output_file = "/data/oceanus_share/shangshouduo-jk/project/datasets/Video-MME/Video-MME_only_question.tsv"

# 读取 TSV 文件
df = pd.read_csv(input_file, sep='\t')

# 去除 question 字段中的固定前缀
df['question'] = df['question'].str.replace(prefix, '', regex=False)

# 保存为新的 TSV 文件
df.to_csv(output_file, sep='\t', index=False)

print(f"处理完成，已保存至 {output_file}")