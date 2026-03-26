import os
import json
import ast
import argparse
import pandas as pd


def options_to_json_str(x):
    """
    VLMEvalKit VideoMMMU expects: line["options"] is a JSON string, e.g. '["a","b","c"]'.
    Your JSONL may store:
      - python-list string with single quotes: "['a','b']"
      - json string: '["a","b"]'
      - python list: ['a','b']
    """
    if x is None:
        return None

    # already a list
    if isinstance(x, list):
        return json.dumps(x, ensure_ascii=False)

    # string: try json.loads first, then ast.literal_eval
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None

        # try JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass

        # try python literal list
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass

    # fallback: cannot parse
    raise ValueError(f"Cannot parse options field into a list: {type(x)} {repr(x)[:200]}")


def ensure_prediction_field(df: pd.DataFrame):
    """
    VLMEvalKit expects 'prediction' column.
    If missing, fallback to 'original_response' if present.
    """
    if "prediction" in df.columns:
        return df
    if "original_response" in df.columns:
        df["prediction"] = df["original_response"]
        return df
    raise ValueError("Input file must contain 'prediction' or 'original_response'.")


def ensure_index_field(df: pd.DataFrame):
    """
    VLMEvalKit VideoMMMU.evaluate uses 'index' as the row key.
    """
    if "index" not in df.columns:
        df["index"] = list(range(len(df)))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True, help="Your inference result jsonl")
    parser.add_argument("--nproc", type=int, default=4, help="VLMEvalKit evaluate nproc")
    parser.add_argument("--out_tsv", default=None, help="Optional: path to write converted TSV")
    args = parser.parse_args()

    in_path = args.input_jsonl
    df = pd.read_json(in_path, lines=True)

    # required fields for VideoMMMU.evaluate:
    # index, id, category, question, options, answer, question_type, prediction
    df = ensure_index_field(df)
    df = ensure_prediction_field(df)

    # Some files may not have 'id' (but your sample has). If missing, reuse index.
    if "id" not in df.columns:
        df["id"] = df["index"]

    # Ensure category exists
    if "category" not in df.columns:
        df["category"] = "Overall"

    # Convert options to JSON string required by VLMEvalKit
    if "options" in df.columns:
        df["options"] = df["options"].apply(options_to_json_str)
    else:
        raise ValueError("Input file must contain 'options' for VideoMMMU.")

    # If your question_type has other variants, normalize here if needed
    # VLMEvalKit handles: multiple-choice, perception, open
    df["question_type"] = df.get("question_type", "open")

    # Keep only the columns VLMEvalKit uses (extra columns are ok, but keep it clean)
    keep_cols = [
        "index", "id", "category", "question", "options", "answer", "question_type", "prediction"
    ]
    for c in keep_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df_out = df[keep_cols].copy()

    # Write TSV for VLMEvalKit evaluate()
    if args.out_tsv is None:
        base = os.path.splitext(in_path)[0]
        out_tsv = base + "_vlmeval.tsv"
    else:
        out_tsv = args.out_tsv

    os.makedirs(os.path.dirname(out_tsv) or ".", exist_ok=True)
    df_out.to_csv(out_tsv, sep="\t", index=False)
    print(f"[OK] Converted TSV saved to: {out_tsv}")

    # ---- Call VLMEvalKit native evaluation chain ----
    # Import path may differ depending on how you installed VLMEvalKit.
    VideoMMMU = None
    import_errors = []

    for mod_path in [
        "vlmeval.dataset",              # some installs expose datasets here
        "vlmeval.datasets",             # alternative
        "vlmeval.dataset.video_mmmu",   # direct module
        "vlmeval.datasets.video_mmmu",
    ]:
        try:
            mod = __import__(mod_path, fromlist=["VideoMMMU"])
            if hasattr(mod, "VideoMMMU"):
                VideoMMMU = getattr(mod, "VideoMMMU")
                break
        except Exception as e:
            import_errors.append((mod_path, str(e)))

    if VideoMMMU is None:
        msg = "\n".join([f"- {p}: {e}" for p, e in import_errors])
        raise ImportError(
            "Cannot import VideoMMMU from VLMEvalKit. Tried:\n"
            f"{msg}\n\n"
            "Make sure you run this script with VLMEvalKit in PYTHONPATH, e.g.\n"
            "PYTHONPATH=/path/to/VLMEvalKit python eval_videommmu_with_vlmevalkit.py --input_jsonl ...\n"
        )

    score_df = VideoMMMU.evaluate(out_tsv, nproc=args.nproc)
    print("\n[VLMEvalKit] Score DataFrame:")
    print(score_df)


if __name__ == "__main__":
    main()
# python /data/oceanus_share/shangshouduo-jk/project/code/refine/evaluate/eval_videommmu_with_vlmevalkit.py --input_jsonl /data/oceanus_share/shangshouduo-jk/project/results/refine/Qwen3-VL-8B-Instruct/VideoMMMU/T20260121_G85738573/Qwen3-VL-8B-Instruct_VideoMMMU.jsonl --nproc 8
# [VLMEvalKit] Score DataFrame:
#        Adaptation  Comprehension  Perception     Overall
# total  300.000000     300.000000  300.000000  900.000000
# hit    142.000000     211.000000  238.000000  591.000000
# acc     47.333333      70.333333   79.333333   65.666667
# python /data/oceanus_share/shangshouduo-jk/myproject/src/evaluate_video/datasets/eval_videommmu_with_vlmevalkit.py --input_jsonl /data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/aks/clip/Qwen3-VL-8B-Instruct/VideoMMMU/T20260205_G85738573/Qwen3-VL-8B-Instruct_VideoMMMU.jsonl --nproc 8
# [VLMEvalKit] Score DataFrame:
#        Adaptation  Comprehension  Perception     Overall
# total  300.000000          300.0  300.000000  900.000000
# hit    169.000000          219.0  248.000000  636.000000
# acc     56.333333           73.0   82.666667   70.666667
# python /data/oceanus_share/shangshouduo-jk/myproject/src/evaluate_video/datasets/eval_videommmu_with_vlmevalkit.py --input_jsonl /data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/q-frame/longclip/Qwen3-VL-8B-Instruct/VideoMMMU/T20260205_G85738573/Qwen3-VL-8B-Instruct_VideoMMMU.jsonl --nproc 8
# [VLMEvalKit] Score DataFrame:
#        Adaptation  Comprehension  Perception     Overall
# total       300.0          300.0       300.0  900.000000
# hit         156.0          198.0       216.0  570.000000
# acc          52.0           66.0        72.0   63.333333