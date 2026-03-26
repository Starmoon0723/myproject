import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, 
                        default='/data/oceanus_share/shangshouduo-jk/project/datasets/MVLU/MLVU/json_new_answer')
    parser.add_argument('--video_path', type=str,
                        default='/data/oceanus_share/shangshouduo-jk/project/datasets/MVLU/MLVU/video')
    parser.add_argument('--output_json', type=str,
                        default='/data/oceanus_share/shangshouduo-jk/myproject/data/processed/MLVU/mlvu_all.json')
    args = parser.parse_args()

    # 定义子任务顺序和名称（与文件名对应）
    subtasks = [
        '1_plotQA',
        '2_needle',
        '3_ego',
        '4_count',
        '5_order',
        '6_anomaly_reco',
        '7_topic_reasoning'
    ]

    all_datas = []

    for subtask in subtasks:
        label_file = os.path.join(args.label_path, f'{subtask}.json')
        video_subdir = os.path.join(args.video_path, subtask)

        if not os.path.exists(label_file):
            print(f"Warning: {label_file} not found, skipping.")
            continue

        with open(label_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        for item in data_list:
            # 添加 video_path 字段：video_subdir + '/' + item['video']
            item['video_path'] = os.path.join(video_subdir, item['video'])
            # 可选：确保 question_type 与子任务一致（去除前缀数字）
            # 当前示例中 question_type 是 'plotQA'，而子任务是 '1_plotQA'
            # 所以我们提取纯名字作为 question_type（如果原数据没有或需统一）
            if 'question_type' not in item or item['question_type'] == "":
                # 从 subtask 名称中提取（去掉数字和下划线）
                clean_type = '_'.join(subtask.split('_')[1:]) if '_' in subtask else subtask
                item['question_type'] = clean_type

            all_datas.append(item)

    # 保存合并后的数据
    with open(args.output_json, 'w', encoding='utf-8') as out_f:
        json.dump(all_datas, out_f, indent=2, ensure_ascii=False)

    print(f"✅ Merged {len(all_datas)} samples into {args.output_json}")

if __name__ == '__main__':
    main()