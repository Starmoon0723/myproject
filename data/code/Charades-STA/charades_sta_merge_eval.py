import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def to_float_pair(x):
    """Convert various timestamp formats to (start, end) floats or (None, None)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None, None

    # list/tuple/np.ndarray
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) >= 2:
            try:
                s = float(x[0])
                e = float(x[1])
                if s > e:
                    s, e = e, s
                return s, e
            except Exception:
                return None, None
        return None, None

    # string like "[24.3, 30.4]" or "24.3 - 30.4"
    if isinstance(x, str):
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", x)
        if len(nums) >= 2:
            try:
                s = float(nums[0])
                e = float(nums[1])
                if s > e:
                    s, e = e, s
                return s, e
            except Exception:
                return None, None
        return None, None

    return None, None


def extract_spans_from_text(text: str):
    """
    Extract multiple (start, end) spans from model output text.
    Strategy: extract all numbers, group into pairs in order.
    """
    if text is None:
        return []
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", str(text))
    if len(nums) < 2:
        return []

    spans = []
    # pair them sequentially: (0,1), (2,3), ...
    for i in range(0, len(nums) - 1, 2):
        try:
            s = float(nums[i])
            e = float(nums[i + 1])
        except Exception:
            continue
        if s > e:
            s, e = e, s
        spans.append((s, e))
    return spans


def iou(seg_p, seg_g):
    ps, pe = seg_p
    gs, ge = seg_g
    if ps is None or pe is None or gs is None or ge is None:
        return 0.0
    if pe < ps:
        ps, pe = pe, ps
    if ge < gs:
        gs, ge = ge, gs

    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    if union <= 0:
        return 0.0
    return float(inter / union)


def safe_py(x):
    """Convert numpy scalars/arrays to JSON-serializable python objects."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_xlsx", required=True, help="Your inference result xlsx")
    ap.add_argument("--gt_parquet", required=True, help="Charades-STA test parquet")
    ap.add_argument("--out_xlsx", default=None, help="Output merged xlsx")
    ap.add_argument("--out_jsonl", default=None, help="Output jsonl with gt+pred")
    ap.add_argument("--prefer_merge_on", choices=["index", "video_caption", "auto"], default="auto")
    args = ap.parse_args()

    pred_xlsx = Path(args.pred_xlsx)
    gt_parquet = Path(args.gt_parquet)

    if args.out_xlsx is None:
        args.out_xlsx = str(pred_xlsx.with_name(pred_xlsx.stem + "_with_gt.xlsx"))
    if args.out_jsonl is None:
        args.out_jsonl = str(pred_xlsx.with_name(pred_xlsx.stem + "_with_gt.jsonl"))

    # 1) load
    df_pred = pd.read_excel(pred_xlsx)
    df_gt = pd.read_parquet(gt_parquet)

    # 2) make sure gt has an index column aligned with row order
    # parquet sample shows no explicit index column; we create row_id as global index
    df_gt = df_gt.reset_index(drop=True)
    df_gt["row_index"] = np.arange(len(df_gt), dtype=np.int64)

    # 3) choose merge strategy
    has_pred_index = "index" in df_pred.columns
    # pred index might be int/float; normalize if possible
    if has_pred_index:
        try:
            df_pred["index"] = df_pred["index"].astype(np.int64)
        except Exception:
            pass

    merge_mode = args.prefer_merge_on
    if merge_mode == "auto":
        # If pred has index and it looks like within gt range, try index first
        if has_pred_index:
            # Heuristic: if max index < len(gt), likely row index
            try:
                if int(df_pred["index"].max()) < len(df_gt):
                    merge_mode = "index"
                else:
                    merge_mode = "video_caption"
            except Exception:
                merge_mode = "video_caption"
        else:
            merge_mode = "video_caption"

    if merge_mode == "index":
        # merge pred.index to gt.row_index
        df_merged = df_pred.merge(
            df_gt[["row_index", "video", "caption", "timestamp"]],
            left_on="index",
            right_on="row_index",
            how="left",
            suffixes=("", "_gt"),
        )
        miss = int(df_merged["timestamp"].isna().sum()) if "timestamp" in df_merged.columns else len(df_merged)
        if miss > 0:
            print(f"[WARN] merge on index: {miss} rows missing GT timestamp. "
                  f"Maybe your pred 'index' is not parquet row index. Consider --prefer_merge_on video_caption.")
    else:
        # merge on (video, caption)
        # detect duplicates in gt keys
        dup_cnt = int(df_gt.duplicated(subset=["video", "caption"]).sum())
        if dup_cnt > 0:
            print(f"[WARN] GT has {dup_cnt} duplicate (video, caption) pairs. "
                  f"Merge may be ambiguous; will keep the first match by default.")
            df_gt_keyed = df_gt.drop_duplicates(subset=["video", "caption"], keep="first")
        else:
            df_gt_keyed = df_gt

        df_merged = df_pred.merge(
            df_gt_keyed[["video", "caption", "timestamp"]],
            on=["video", "caption"],
            how="left",
            suffixes=("", "_gt"),
        )
        miss = int(df_merged["timestamp"].isna().sum()) if "timestamp" in df_merged.columns else len(df_merged)
        if miss > 0:
            print(f"[WARN] merge on (video, caption): {miss} rows missing GT timestamp. "
                  f"These rows will get (None, None) gt span and IoU=0.")

    # 4) parse GT timestamp -> gt_start/gt_end
    gt_starts, gt_ends = [], []
    for x in df_merged.get("timestamp", pd.Series([None] * len(df_merged))):
        s, e = to_float_pair(x)
        gt_starts.append(s)
        gt_ends.append(e)
    df_merged["gt_start"] = gt_starts
    df_merged["gt_end"] = gt_ends
    df_merged["gt_timestamp"] = df_merged.apply(
        lambda r: [r["gt_start"], r["gt_end"]] if (r["gt_start"] is not None and r["gt_end"] is not None) else None,
        axis=1
    )

    # 5) parse prediction -> spans
    pred_first_s, pred_first_e = [], []
    pred_best_s, pred_best_e = [], []
    iou_first_list, iou_best_list = [], []
    n_spans_list = []

    for _, r in df_merged.iterrows():
        spans = extract_spans_from_text(r.get("prediction", None))
        n_spans_list.append(len(spans))

        gs, ge = r.get("gt_start", None), r.get("gt_end", None)
        gt_seg = (gs, ge)

        # first span
        if len(spans) >= 1:
            fs, fe = spans[0]
        else:
            fs, fe = (None, None)
        pred_first_s.append(fs)
        pred_first_e.append(fe)
        iou_first_list.append(iou((fs, fe), gt_seg))

        # best span by IoU
        if len(spans) >= 1 and (gs is not None and ge is not None):
            best = max(spans, key=lambda sp: iou(sp, gt_seg))
            bs, be = best
            best_i = max(iou(sp, gt_seg) for sp in spans)
        else:
            bs, be = (None, None)
            best_i = 0.0
        pred_best_s.append(bs)
        pred_best_e.append(be)
        iou_best_list.append(best_i)

    df_merged["pred_n_spans"] = n_spans_list
    df_merged["pred_first_start"] = pred_first_s
    df_merged["pred_first_end"] = pred_first_e
    df_merged["pred_best_start"] = pred_best_s
    df_merged["pred_best_end"] = pred_best_e
    df_merged["iou_first"] = iou_first_list
    df_merged["iou_best"] = iou_best_list

    # 6) metrics
    miou_first = float(np.mean(df_merged["iou_first"].values)) if len(df_merged) else 0.0
    miou_best = float(np.mean(df_merged["iou_best"].values)) if len(df_merged) else 0.0
    valid_first = float(np.mean(df_merged["pred_first_start"].notna().values)) if len(df_merged) else 0.0
    valid_best = float(np.mean(df_merged["pred_best_start"].notna().values)) if len(df_merged) else 0.0

    metrics = {
        "dataset": "Charades-STA",
        "n_samples": int(len(df_merged)),
        "mIoU_first": miou_first,
        "mIoU_best": miou_best,
        "valid_pred_rate_first": valid_first,
        "valid_pred_rate_best": valid_best,
        "merge_mode": merge_mode,
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # 7) write merged xlsx
    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_excel(out_xlsx, index=False)
    print(f"[OK] Wrote merged xlsx: {out_xlsx}")

    # 8) write jsonl
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for _, r in df_merged.iterrows():
            record = {
                "index": safe_py(r.get("index", None)),
                "video": safe_py(r.get("video", None)),
                "caption": safe_py(r.get("caption", None)),
                "video_path": safe_py(r.get("video_path", None)),
                "gt_timestamp": safe_py(r.get("gt_timestamp", None)),
                "gt_start": safe_py(r.get("gt_start", None)),
                "gt_end": safe_py(r.get("gt_end", None)),
                "prediction_text": safe_py(r.get("prediction", None)),
                "original_response": safe_py(r.get("original_response", None)),
                "pred_first": [safe_py(r.get("pred_first_start", None)), safe_py(r.get("pred_first_end", None))] \
                              if pd.notna(r.get("pred_first_start", np.nan)) else None,
                "pred_best": [safe_py(r.get("pred_best_start", None)), safe_py(r.get("pred_best_end", None))] \
                             if pd.notna(r.get("pred_best_start", np.nan)) else None,
                "pred_n_spans": safe_py(r.get("pred_n_spans", None)),
                "iou_first": safe_py(r.get("iou_first", None)),
                "iou_best": safe_py(r.get("iou_best", None)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote jsonl: {out_jsonl}")


if __name__ == "__main__":
    main()

# python /data/oceanus_share/shangshouduo-jk/project/datasets/Charades-STA/charades_sta_merge_eval.py --pred_xlsx /data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/Charades-STA/T20260115_G85738573_old/Qwen3-VL-8B-Instruct_Charades-STA.xlsx --gt_parquet /data/oceanus_share/shangshouduo-jk/project/datasets/Charades-STA/data/test-00000-of-00001.parquet --prefer_merge_on index
# python /data/oceanus_share/shangshouduo-jk/project/datasets/Charades-STA/charades_sta_merge_eval.py --pred_xlsx /data/oceanus_share/shangshouduo-jk/project/results/Qwen3-VL-8B-Instruct/Charades-STA/T20260114_G85738573/Qwen3-VL-8B-Instruct_Charades-STA.xlsx --gt_parquet /data/oceanus_share/shangshouduo-jk/project/datasets/Charades-STA/data/test-00000-of-00001.parquet --prefer_merge_on index
# {
#   "dataset": "Charades-STA",
#   "n_samples": 3720,
#   "mIoU_first": 0.5153636747570096,
#   "mIoU_best": 0.5153636747570096,
#   "valid_pred_rate_first": 0.9994623655913979,
#   "valid_pred_rate_best": 0.9994623655913979,
#   "merge_mode": "index"
# }