import heapq
import json
import numpy as np
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')

    parser.add_argument('--dataset_name', type=str, default='videomme', help='support longvideobench and videomme')
    parser.add_argument('--extract_feature_model', type=str, default='clip', help='blip/clip/sevila')
    parser.add_argument('--score_path', type=str, default='/data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/aks/AKS/outscores/videomme/clip/scores.json')
    parser.add_argument('--frame_path', type=str, default='/data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/aks/AKS/outscores/videomme/clip/frames.json')
    parser.add_argument('--max_num_frames', type=int, default=64)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--t1', type=float, default=0.8)
    parser.add_argument('--t2', type=float, default=-100)
    parser.add_argument('--all_depth', type=int, default=5)
    parser.add_argument('--output_file', type=str, default='/data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/aks/code/AKS/selected_frames')

    return parser.parse_args()

def meanstd(len_scores, dic_scores, n, fns,t1,t2,all_depth):
        split_scores = []
        split_fn = []
        no_split_scores = []
        no_split_fn = []
        i= 0
        for dic_score, fn in zip(dic_scores, fns):
                # normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
                score = dic_score['score']
                depth = dic_score['depth']
                mean = np.mean(score)
                std = np.std(score)

                top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
                top_score = [score[t] for t in top_n]
                # print(f"split {i}: ",len(score))
                i += 1
                mean_diff = np.mean(top_score) - mean
                if mean_diff > t1 and std > t2:
                        no_split_scores.append(dic_score)
                        no_split_fn.append(fn)
                elif depth < all_depth:
                # elif len(score)>(len_scores/n)*2 and len(score) >= 8:
                        score1 = score[:len(score)//2]
                        score2 = score[len(score)//2:]
                        fn1 = fn[:len(score)//2]
                        fn2 = fn[len(score)//2:]                       
                        split_scores.append(dict(score=score1,depth=depth+1))
                        split_scores.append(dict(score=score2,depth=depth+1))
                        split_fn.append(fn1)
                        split_fn.append(fn2)
                else:
                        no_split_scores.append(dic_score)
                        no_split_fn.append(fn)
        if len(split_scores) > 0:
                all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn,t1,t2,all_depth)
        else:
                all_split_score = []
                all_split_fn = []
        all_split_score = no_split_scores + all_split_score
        all_split_fn = no_split_fn + all_split_fn


        return all_split_score, all_split_fn

def load_json_or_jsonl(path):
    """
    Load a file that is either:
      - A standard JSON list of lists: [[...], [...]]
      - A JSONL file where each line is a JSON list: [...] \n [...] \n ...
    Returns a list of lists.
    """
    with open(path, 'r') as f:
        content = f.read().strip()
    
    if not content:
        return []

    try:
        # Try parsing as standard JSON
        data = json.loads(content)
        if isinstance(data, list):
            # If it's a list of numbers (not list of lists), assume single video
            if len(data) > 0 and not isinstance(data[0], list):
                return [data]
            else:
                return data
        else:
            raise ValueError("Root element is not a list")
    except (json.JSONDecodeError, ValueError, IndexError):
        # Fall back to JSONL: split by lines and parse each
        data = []
        for line_num, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    if isinstance(item, list):
                        data.append(item)
                    else:
                        print(f"Warning: Line {line_num} in {path} is not a list; skipped.")
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON on line {line_num} in {arg}; skipped.")
        return data

def main(args):
    max_num_frames = args.max_num_frames
    ratio = args.ratio
    t1 = args.t1
    t2 = args.t2
    all_depth = args.all_depth
    outs = []
    segs = []

    # with open(args.score_path) as f:
    #     itm_outs = json.load(f)
    # with open(args.frame_path) as f:
    #     fn_outs = json.load(f)
    itm_outs = load_json_or_jsonl(args.score_path)
    fn_outs = load_json_or_jsonl(args.frame_path)

    # print(itm_outs)

    if not os.path.exists(os.path.join(args.output_file,args.dataset_name)):
        os.mkdir(os.path.join(args.output_file,args.dataset_name))
    out_score_path = os.path.join(args.output_file,args.dataset_name,args.extract_feature_model)
    if not os.path.exists(out_score_path):
        os.mkdir(out_score_path)

    for itm_out,fn_out in zip(itm_outs,fn_outs):
        nums = int(len(itm_out)/ratio)
        new_score = [itm_out[num*ratio] for num in range(nums)]
        new_fnum = [fn_out[num*ratio] for num in range(nums)]
        score = new_score
        fn = new_fnum
        num = max_num_frames
        if len(score) >= num:
            normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
            a, b = meanstd(len(score), [dict(score=normalized_data,depth=0)], num, [fn], t1, t2, all_depth)
            segs.append(len(a))
            out = []
            if len(score) >= num:
                for s,f in zip(a,b): 
                    f_num = int(num / 2**(s['depth']))
                    topk = heapq.nlargest(f_num, range(len(s['score'])), s['score'].__getitem__)
                    f_nums = [f[t] for t in topk]
                    out.extend(f_nums)
            out.sort()
            outs.append(out)
        else:
            outs.append(fn)

    score_path = os.path.join(out_score_path,'selected_frames.json')
    with open(score_path,'w') as f:
        json.dump(outs,f)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)