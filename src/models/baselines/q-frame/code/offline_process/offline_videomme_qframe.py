import argparse
import ast
import json
import os
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def _extract_question_text(raw_question: str) -> str:
    """
    Video-MME_copy.tsv 的 question 字段通常包含类似：
      "... Question: <real question>"
    q-frame 的 TextImageMatching(videomme) 实际只取 question 这一行。
    这里做一个稳健抽取：优先截取最后一个 "Question:" 后面的内容。
    """
    if raw_question is None:
        return ""
    s = str(raw_question).strip()
    if "Question:" in s:
        s = s.split("Question:")[-1].strip()
    return s


def _parse_candidates(raw: str) -> List[str]:
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return [s]


def _load_longclip(longclip_root: Path, ckpt_path: Path, device: str):
    import sys

    longclip_root = longclip_root.resolve()
    if str(longclip_root) not in sys.path:
        sys.path.insert(0, str(longclip_root))
    # Long-CLIP repo convention (matches q-frame upstream)
    from model import longclip  # type: ignore

    clip_model, clip_processor = longclip.load(str(ckpt_path), device=device)
    clip_model.eval()
    return longclip, clip_model, clip_processor


def _text_image_matching(
    longclip_mod,
    clip_model,
    clip_processor,
    question: str,
    frames_np: np.ndarray,
    device: str,
    tau: float = 0.8,
    use_gumbel: bool = True,
) -> np.ndarray:
    """
    复刻 q-frame 的核心打分：
      logits -> softmax/tau -> (可选) Gumbel 噪声 -> 排序
    返回的是对 frames_np 维度 0 的“位置索引”排序（降序）。
    """
    if len(frames_np) == 0:
        return np.array([], dtype=np.int64)

    with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=(device.startswith("cuda"))):
        text = longclip_mod.tokenize([question]).to(device)
        images = torch.stack([clip_processor(Image.fromarray(im)) for im in frames_np]).to(device)
        image_features = clip_model.encode_image(images)
        text_features = clip_model.encode_text(text)
        logits_per_text = text_features @ image_features.T  # (1, T)
        probs = (logits_per_text / float(tau)).softmax(dim=1)[0]  # (T,)

        if use_gumbel:
            # q-frame: log(p) - log(-log(U))
            u = torch.rand(len(frames_np), device=probs.device) + 1e-10
            g = -torch.log(u) + 1e-10
            g = -torch.log(g) + 1e-10
            scores = torch.log(probs + 1e-10) - torch.log(g)
        else:
            scores = probs

    return np.argsort(-scores.detach().float().cpu().numpy())


def _load_video_candidates_decord(
    video_path: str,
    max_frames_num: int,
    target_fps: float = 1.0,
) -> Tuple[np.ndarray, List[int]]:
    import decord

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    total = len(vr)
    avg_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else 0.0
    if avg_fps <= 0:
        # fallback: treat as 1fps to avoid div0
        avg_fps = 1.0

    stride = max(1, int(round(avg_fps / float(target_fps))))
    frame_idx = list(range(0, total, stride))

    if len(frame_idx) > max_frames_num:
        uniform = np.linspace(0, total - 1, max_frames_num, dtype=int).tolist()
        frame_idx = uniform

    frames = vr.get_batch(frame_idx).asnumpy()  # (T,H,W,3)
    return frames, frame_idx


@dataclass
class QFrameParams:
    max_frames_num: int = 128
    target_fps: float = 1.0
    high_frames: int = 4
    mid_frames: int = 8
    low_frames: int = 32
    tau: float = 0.8


def _save_multires_images(
    frames_np: np.ndarray,
    candidate_frame_indices: List[int],
    ranked_positions: np.ndarray,
    out_dir: Path,
    params: QFrameParams,
    image_ext: str = "jpg",
) -> Tuple[List[str], List[int], List[int]]:
    """
    输出：
    - image_paths: 按“时间顺序（candidate position）”排列的多图路径列表
    - selected_frame_indices: 对应原视频的 frame index（与 image_paths 一一对应）
    - selected_divs: 每张图对应的缩放倍率分母（2/4/8）
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    hi = set(int(x) for x in ranked_positions[: params.high_frames].tolist())
    mid = set(int(x) for x in ranked_positions[params.high_frames : params.high_frames + params.mid_frames].tolist())
    low = set(
        int(x)
        for x in ranked_positions[
            params.high_frames + params.mid_frames : params.high_frames + params.mid_frames + params.low_frames
        ].tolist()
    )
    selected_pos = sorted(list(hi | mid | low))

    if len(selected_pos) == 0:
        return [], [], []

    # base size from original frame (like upstream q-frame)
    pil0 = Image.fromarray(frames_np[0]).convert("RGB")
    w0, h0 = pil0.size

    image_paths: List[str] = []
    selected_frame_indices: List[int] = []
    selected_divs: List[int] = []

    for pos in selected_pos:
        if pos in hi:
            div = 2
        elif pos in mid:
            div = 4
        else:
            div = 8

        orig_fi = int(candidate_frame_indices[pos])
        img = Image.fromarray(frames_np[pos]).convert("RGB")
        new_w = max(1, w0 // div)
        new_h = max(1, h0 // div)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        fn = f"f{orig_fi:06d}_p{pos:03d}_d{div}.{image_ext}"
        fp = out_dir / fn
        img.save(fp, quality=95)

        image_paths.append(str(fp))
        selected_frame_indices.append(orig_fi)
        selected_divs.append(div)

    return image_paths, selected_frame_indices, selected_divs

def load_input_data(path: Path):
    """根据后缀名动态加载数据"""
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 如果是 JSON，通常是 List[Dict]，直接转成 DataFrame 比较方便统一处理
        return pd.DataFrame(data)
    elif suffix in [".tsv", ".csv"]:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

# 在 main 函数中替换原有的 pd.read_csv：
# df = load_input_data(tsv_path)

def main():
    parser = argparse.ArgumentParser(description="Offline preprocessing for q-frame on Video-MME (multi-res image list).")
    parser.add_argument(
        "--tsv_path",
        type=str,
        default="/data/oceanus_share/shangshouduo-jk/project/datasets/Video-MME/Video-MME_only_question.tsv",
        help="Path to Video-MME_copy.tsv (ordered).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/data/oceanus_share/shangshouduo-jk/myproject/data/processed/Video-MME/baselines/q-frame",
        help="Output root dir, will create manifests + images here.",
    )
    parser.add_argument(
        "--longclip_root",
        type=str,
        default="/data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/q-frame/Long-CLIP",
        help="Path to Long-CLIP repo root.",
    )
    parser.add_argument(
        "--longclip_ckpt",
        type=str,
        default="/data/oceanus_share/shangshouduo-jk/myproject/ckpts/longclip/longclip-L.pt",
        help="Checkpoint path under longclip_root.",
    )
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--resume", action="store_true", help="Resume by skipping already written lines (by order).")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N rows (0 means all).")
    parser.add_argument("--image_ext", type=str, default="jpg")

    # q-frame params (match experiments/videomme/qwen2vl/qframe.sh intent)
    parser.add_argument("--max_frames_num", type=int, default=128)
    parser.add_argument("--target_fps", type=float, default=1.0)
    parser.add_argument("--high_frames", type=int, default=4)
    parser.add_argument("--mid_frames", type=int, default=8)
    parser.add_argument("--low_frames", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--no_gumbel", action="store_true", help="Disable Gumbel noise for deterministic ranking.")

    args = parser.parse_args()

    tsv_path = Path(args.tsv_path)
    out_root = Path(args.output_root)
    longclip_root = Path(args.longclip_root)
    ckpt_path = longclip_root / args.longclip_ckpt

    out_root.mkdir(parents=True, exist_ok=True)
    images_root = out_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    cand_jsonl = out_root / "candidate_frame_indices.jsonl"
    sel_jsonl = out_root / "selected_frame_indices.jsonl"
    manifest_jsonl = out_root / "selected_frames_manifest.jsonl"

    start_idx = 0
    if args.resume:
        # 以 manifest 为准（因为评估主要用它）
        start_idx = _count_jsonl_lines(manifest_jsonl)

    params = QFrameParams(
        max_frames_num=args.max_frames_num,
        target_fps=args.target_fps,
        high_frames=args.high_frames,
        mid_frames=args.mid_frames,
        low_frames=args.low_frames,
        tau=args.tau,
    )

    df = load_input_data(tsv_path)
    # df = pd.read_csv(tsv_path, sep="\t")
    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].reset_index(drop=True)

    longclip_mod, clip_model, clip_processor = _load_longclip(longclip_root, ckpt_path, args.device)

    total = len(df)
    for row_i in range(start_idx, total):
        row = df.iloc[row_i].to_dict()
        sample_index = int(row.get("index", row_i))
        video_path = str(row["video_path"])
        raw_q = str(row.get("question", ""))
        q_text = _extract_question_text(raw_q)
        candidates = _parse_candidates(row.get("candidates", ""))
        answer = str(row.get("answer", ""))

        # Candidate frames
        frames_np, candidate_fi = _load_video_candidates_decord(
            video_path=video_path,
            max_frames_num=params.max_frames_num,
            target_fps=params.target_fps,
        )

        _append_jsonl(
            cand_jsonl,
            dict(
                index=sample_index,
                video_path=video_path,
                candidate_frame_indices=candidate_fi,
                num_candidates=len(candidate_fi),
            ),
        )

        # Ranking (q-frame uses question-only for videomme)
        ranked_pos = _text_image_matching(
            longclip_mod=longclip_mod,
            clip_model=clip_model,
            clip_processor=clip_processor,
            question=q_text,
            frames_np=frames_np,
            device=args.device,
            tau=params.tau,
            use_gumbel=(not args.no_gumbel),
        )

        # Save multi-res images
        sample_img_dir = images_root / f"{sample_index:06d}"
        image_paths, selected_fi, selected_divs = _save_multires_images(
            frames_np=frames_np,
            candidate_frame_indices=candidate_fi,
            ranked_positions=ranked_pos,
            out_dir=sample_img_dir,
            params=params,
            image_ext=args.image_ext,
        )

        _append_jsonl(
            sel_jsonl,
            dict(
                index=sample_index,
                video_path=video_path,
                selected_frame_indices=selected_fi,
                selected_divs=selected_divs,
            ),
        )

        _append_jsonl(
            manifest_jsonl,
            dict(
                index=sample_index,
                video_path=video_path,
                question=raw_q,
                candidates=candidates,
                answer=answer,
                candidate_frame_indices=candidate_fi,
                selected_frame_indices=selected_fi,
                selected_divs=selected_divs,
                image_paths=image_paths,  # 评估侧建议直接用这个（多张 image 输入）
                qframe_params=dict(
                    max_frames_num=params.max_frames_num,
                    target_fps=params.target_fps,
                    high_frames=params.high_frames,
                    mid_frames=params.mid_frames,
                    low_frames=params.low_frames,
                    tau=params.tau,
                    use_gumbel=(not args.no_gumbel),
                ),
            ),
        )

        if (row_i + 1) % 10 == 0 or row_i == total - 1:
            print(f"[q-frame offline] processed {row_i+1}/{total}")


if __name__ == "__main__":
    main()

# python /data/oceanus_share/shangshouduo-jk/myproject/src/models/baselines/q-frame/code/offline_videomme_qframe.py --resume