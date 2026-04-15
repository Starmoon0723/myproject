#!/usr/bin/env python3
"""
Codec-domain frame proposal for compressed videos with I/P frames only.

Goal:
- Never dump all frames to images.
- Read frame type / packet-size metadata via ffprobe.
- Read block-level motion vectors directly via PyAV/FFmpeg side-data.
- Score and select frames using:
    1) forced I-frame anchors
    2) boundary spikes
    3) accumulated novelty
    4) max-gap fallback

Why packet size is used:
- FFmpeg/ffprobe can expose frame types and packet sizes very cheaply.
- Stock ffprobe usually does NOT emit per-block residual tensors. For a first
  compressed-domain experiment, P-frame packet size is used as a strong,
  cheap residual proxy.

Why PyAV is used for motion vectors:
- FFmpeg decoder can export motion vectors as frame side-data.
- PyAV can read that side-data directly in Python, so no external extract_mvs
  binary or extra compilation step is required.

Typical usage:
python codec_frame_sampler.py \
  --input_dir /data/oceanus_ctr/j-shangshouduo-jk/project/data/raw/videomme/data_fps2 \
  --output_dir /data/oceanus_ctr/j-shangshouduo-jk/project/data/raw/videomme/codec_sampling_v1 \
  --workers 8 \
  --target_kept_per_gop 4.0 \
  --max_gap 4 \
  --boundary_quantile 0.92
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import av
except Exception:
    av = None


# -----------------------------
# Data classes
# -----------------------------


@dataclass
class FrameInfo:
    frame_idx0: int
    pts_time: float
    pict_type: str
    key_frame: int
    pkt_size: int
    width: int
    height: int
    gop_id: int = -1
    within_gop_idx: int = -1


@dataclass
class MotionVector:
    frame_num1: int
    source: int
    block_w: int
    block_h: int
    src_x: int
    src_y: int
    dst_x: int
    dst_y: int
    flags: str
    motion_x: int
    motion_y: int
    motion_scale: int

    @property
    def area(self) -> float:
        return float(max(self.block_w, 1) * max(self.block_h, 1))

    @property
    def dx(self) -> float:
        # Keep the original code logic unchanged:
        # use geometric displacement from source to destination.
        return float(self.dst_x - self.src_x)

    @property
    def dy(self) -> float:
        return float(self.dst_y - self.src_y)


@dataclass
class FrameFeatures:
    frame_idx0: int
    pts_time: float
    pict_type: str
    gop_id: int
    within_gop_idx: int
    pkt_size: int
    pkt_log: float
    pkt_jump: float
    mv_count: int = 0
    mv_mean: float = 0.0
    mv_p90: float = 0.0
    mv_active_ratio: float = 0.0
    dir_entropy: float = 0.0
    global_dx: float = 0.0
    global_dy: float = 0.0
    global_motion_delta: float = 0.0
    novelty_raw: float = 0.0
    boundary_raw: float = 0.0
    keep: bool = False
    keep_reason: str = ""
    score_accum_at_keep: float = 0.0
    debug: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Helpers
# -----------------------------


def run_cmd(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def robust_median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(statistics.median(xs))


def mad(xs: Sequence[float], med: Optional[float] = None) -> float:
    if not xs:
        return 1.0
    if med is None:
        med = robust_median(xs)
    abs_dev = [abs(x - med) for x in xs]
    m = robust_median(abs_dev)
    return m if m > 1e-9 else 1.0


def robust_zscores(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    med = robust_median(values)
    scale = 1.4826 * mad(values, med) + 1e-9
    return [(v - med) / scale for v in values]


def relu(x: float) -> float:
    return x if x > 0 else 0.0


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    assert len(values) == len(weights)
    if not values:
        return 0.0
    q = min(max(q, 0.0), 1.0)
    pairs = sorted(zip(values, weights), key=lambda t: t[0])
    total = sum(max(w, 0.0) for _, w in pairs)
    if total <= 0:
        return float(pairs[len(pairs) // 2][0])
    cutoff = q * total
    csum = 0.0
    for v, w in pairs:
        csum += max(w, 0.0)
        if csum >= cutoff:
            return float(v)
    return float(pairs[-1][0])


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    return weighted_quantile(values, weights, 0.5)


def entropy_from_probs(probs: Sequence[float]) -> float:
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    h = -sum(p * math.log(p + 1e-12) for p in probs)
    hmax = math.log(len(probs)) if len(probs) > 1 else 1.0
    return float(h / (hmax + 1e-12))


def find_videos(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.mp4") if p.is_file()])


def cache_json_path_for_video(cache_dir: Path, input_dir: Path, video_path: Path) -> Path:
    rel = video_path.relative_to(input_dir)
    return cache_dir / rel.with_suffix(".json")


# -----------------------------
# ffprobe / motion-vector parsing
# -----------------------------


def ffprobe_frames(video_path: Path) -> List[FrameInfo]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pict_type,key_frame,pkt_size,width,height",
        "-of",
        "json",
        str(video_path),
    ]
    res = run_cmd(cmd)
    payload = json.loads(res.stdout)
    raw_frames = payload.get("frames", [])

    frames: List[FrameInfo] = []
    gop_id = -1
    within_gop = -1
    for idx0, fr in enumerate(raw_frames):
        pict_type = str(fr.get("pict_type", "?"))
        key_frame = safe_int(fr.get("key_frame", 0))
        pkt_size = safe_int(fr.get("pkt_size", 0))
        pts_time = safe_float(fr.get("best_effort_timestamp_time", idx0))
        width = safe_int(fr.get("width", 0))
        height = safe_int(fr.get("height", 0))
        if pict_type == "I" or key_frame == 1:
            gop_id += 1
            within_gop = 0
        else:
            within_gop += 1
        frames.append(
            FrameInfo(
                frame_idx0=idx0,
                pts_time=pts_time,
                pict_type=pict_type,
                key_frame=key_frame,
                pkt_size=pkt_size,
                width=width,
                height=height,
                gop_id=max(gop_id, 0),
                within_gop_idx=max(within_gop, 0),
            )
        )
    return frames


def find_mv_extractor(explicit: Optional[str]) -> Optional[str]:
    """
    Kept only for backward compatibility with the original CLI/output schema.
    External extract_mvs is no longer used.
    """
    _ = explicit
    return None


def _get_motion_vector_side_data(frame: Any) -> Optional[Any]:
    try:
        for sd in frame.side_data:
            t = getattr(sd, "type", None)
            tname = getattr(t, "name", str(t))
            if "MOTION_VECTORS" in tname:
                return sd
    except Exception:
        pass

    try:
        return frame.side_data.get("MOTION_VECTORS")
    except Exception:
        pass

    return None


def parse_motion_vectors_with_pyav(video_path: Path) -> Dict[int, List[MotionVector]]:
    """
    Returns:
        frame_num1 -> list of MotionVector

    Uses PyAV to read FFmpeg decoder motion-vector side-data directly.
    frame_num1 is 1-based and matches the existing downstream logic:
    build_frame_features() queries frame_idx0 + 1.
    """
    if av is None:
        raise RuntimeError(
            "PyAV is not installed. Please install it first, e.g.:\n"
            "pip install av"
        )

    out: Dict[int, List[MotionVector]] = {}

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        codec_ctx = stream.codec_context

        # Enable motion-vector export from the FFmpeg decoder.
        configured = False
        try:
            codec_ctx.flags2 |= av.codec.context.Flags2.EXPORT_MVS
            configured = True
        except Exception:
            pass

        if not configured:
            try:
                opts = dict(getattr(codec_ctx, "options", {}) or {})
                opts["flags2"] = "+export_mvs"
                codec_ctx.options = opts
                configured = True
            except Exception:
                pass

        try:
            is_open = bool(getattr(codec_ctx, "is_open", False))
            if not is_open:
                codec_ctx.open()
        except Exception:
            pass

        frame_num1 = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_num1 += 1

                mv_side_data = _get_motion_vector_side_data(frame)
                if mv_side_data is None:
                    continue

                mvs: List[MotionVector] = []
                for mv in mv_side_data:
                    motion_scale = safe_int(getattr(mv, "motion_scale", 1), 1)
                    if motion_scale == 0:
                        motion_scale = 1

                    mvs.append(
                        MotionVector(
                            frame_num1=frame_num1,
                            source=safe_int(getattr(mv, "source", 0)),
                            block_w=safe_int(getattr(mv, "w", 0)),
                            block_h=safe_int(getattr(mv, "h", 0)),
                            src_x=safe_int(getattr(mv, "src_x", 0)),
                            src_y=safe_int(getattr(mv, "src_y", 0)),
                            dst_x=safe_int(getattr(mv, "dst_x", 0)),
                            dst_y=safe_int(getattr(mv, "dst_y", 0)),
                            flags=str(getattr(mv, "flags", 0)),
                            motion_x=safe_int(getattr(mv, "motion_x", 0)),
                            motion_y=safe_int(getattr(mv, "motion_y", 0)),
                            motion_scale=motion_scale,
                        )
                    )

                if mvs:
                    out[frame_num1] = mvs

    return out


def save_offline_video_data(
    out_path: Path,
    input_dir: Path,
    video_path: Path,
    frames: List[FrameInfo],
    frame_to_mvs: Optional[Dict[int, List[MotionVector]]],
    meta: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rel = str(video_path.relative_to(input_dir))
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "video_path": str(video_path),
        "video_rel_path": rel,
        "meta": meta,
        "frames": [
            {
                "frame_idx0": f.frame_idx0,
                "pts_time": f.pts_time,
                "pict_type": f.pict_type,
                "key_frame": f.key_frame,
                "pkt_size": f.pkt_size,
                "width": f.width,
                "height": f.height,
                "gop_id": f.gop_id,
                "within_gop_idx": f.within_gop_idx,
            }
            for f in frames
        ],
        "motion_vectors": {},
    }
    if frame_to_mvs is not None:
        payload["motion_vectors"] = {
            str(frame_num1): [
                {
                    "frame_num1": mv.frame_num1,
                    "source": mv.source,
                    "block_w": mv.block_w,
                    "block_h": mv.block_h,
                    "src_x": mv.src_x,
                    "src_y": mv.src_y,
                    "dst_x": mv.dst_x,
                    "dst_y": mv.dst_y,
                    "flags": mv.flags,
                    "motion_x": mv.motion_x,
                    "motion_y": mv.motion_y,
                    "motion_scale": mv.motion_scale,
                }
                for mv in mvs
            ]
            for frame_num1, mvs in frame_to_mvs.items()
        }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_offline_video_data(
    cache_path: Path,
) -> Tuple[List[FrameInfo], Optional[Dict[int, List[MotionVector]]], Dict[str, Any]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))

    raw_frames = payload.get("frames", [])
    frames = [
        FrameInfo(
            frame_idx0=safe_int(fr.get("frame_idx0", 0)),
            pts_time=safe_float(fr.get("pts_time", 0.0)),
            pict_type=str(fr.get("pict_type", "?")),
            key_frame=safe_int(fr.get("key_frame", 0)),
            pkt_size=safe_int(fr.get("pkt_size", 0)),
            width=safe_int(fr.get("width", 0)),
            height=safe_int(fr.get("height", 0)),
            gop_id=safe_int(fr.get("gop_id", 0)),
            within_gop_idx=safe_int(fr.get("within_gop_idx", 0)),
        )
        for fr in raw_frames
    ]

    raw_mvs = payload.get("motion_vectors", None)
    frame_to_mvs: Optional[Dict[int, List[MotionVector]]] = None
    if isinstance(raw_mvs, dict):
        frame_to_mvs = {}
        for k, mv_list in raw_mvs.items():
            frame_num1 = safe_int(k, -1)
            if frame_num1 <= 0:
                continue
            parsed: List[MotionVector] = []
            for mv in mv_list or []:
                parsed.append(
                    MotionVector(
                        frame_num1=safe_int(mv.get("frame_num1", frame_num1), frame_num1),
                        source=safe_int(mv.get("source", 0)),
                        block_w=safe_int(mv.get("block_w", 0)),
                        block_h=safe_int(mv.get("block_h", 0)),
                        src_x=safe_int(mv.get("src_x", 0)),
                        src_y=safe_int(mv.get("src_y", 0)),
                        dst_x=safe_int(mv.get("dst_x", 0)),
                        dst_y=safe_int(mv.get("dst_y", 0)),
                        flags=str(mv.get("flags", "0")),
                        motion_x=safe_int(mv.get("motion_x", 0)),
                        motion_y=safe_int(mv.get("motion_y", 0)),
                        motion_scale=max(safe_int(mv.get("motion_scale", 1), 1), 1),
                    )
                )
            if parsed:
                frame_to_mvs[frame_num1] = parsed

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    return frames, frame_to_mvs, meta


# -----------------------------
# Feature extraction
# -----------------------------


def compute_motion_features(mvs: List[MotionVector], motion_eps: float = 0.75) -> Dict[str, float]:
    if not mvs:
        return {
            "mv_count": 0,
            "mv_mean": 0.0,
            "mv_p90": 0.0,
            "mv_active_ratio": 0.0,
            "dir_entropy": 0.0,
            "global_dx": 0.0,
            "global_dy": 0.0,
        }

    dxs = [mv.dx for mv in mvs]
    dys = [mv.dy for mv in mvs]
    ws = [mv.area for mv in mvs]

    gdx = weighted_median(dxs, ws)
    gdy = weighted_median(dys, ws)

    mags: List[float] = []
    mag_weights: List[float] = []
    angle_bins = [0.0] * 8
    total_area = 0.0
    active_area = 0.0

    for mv, w in zip(mvs, ws):
        local_dx = mv.dx - gdx
        local_dy = mv.dy - gdy
        mag = math.hypot(local_dx, local_dy)
        mags.append(mag)
        mag_weights.append(w)
        total_area += w
        if mag > motion_eps:
            active_area += w
            angle = math.atan2(local_dy, local_dx)  # [-pi, pi]
            bin_id = int(((angle + math.pi) / (2 * math.pi)) * 8) % 8
            angle_bins[bin_id] += w

    mv_mean = sum(m * w for m, w in zip(mags, mag_weights)) / (sum(mag_weights) + 1e-9)
    mv_p90 = weighted_quantile(mags, mag_weights, 0.90)
    mv_active_ratio = active_area / (total_area + 1e-9)
    dir_probs = [b / (sum(angle_bins) + 1e-9) for b in angle_bins]
    dir_entropy = entropy_from_probs(dir_probs)

    return {
        "mv_count": int(len(mvs)),
        "mv_mean": float(mv_mean),
        "mv_p90": float(mv_p90),
        "mv_active_ratio": float(mv_active_ratio),
        "dir_entropy": float(dir_entropy),
        "global_dx": float(gdx),
        "global_dy": float(gdy),
    }


def build_frame_features(
    frames: List[FrameInfo],
    frame_to_mvs: Optional[Dict[int, List[MotionVector]]] = None,
    motion_eps: float = 0.75,
) -> List[FrameFeatures]:
    features: List[FrameFeatures] = []
    prev_pkt_log = 0.0
    prev_global_dx = 0.0
    prev_global_dy = 0.0

    for fr in frames:
        pkt_log = math.log1p(max(fr.pkt_size, 0))
        pkt_jump = abs(pkt_log - prev_pkt_log) if features else 0.0

        motion = {
            "mv_count": 0,
            "mv_mean": 0.0,
            "mv_p90": 0.0,
            "mv_active_ratio": 0.0,
            "dir_entropy": 0.0,
            "global_dx": 0.0,
            "global_dy": 0.0,
        }
        if frame_to_mvs is not None:
            motion = compute_motion_features(frame_to_mvs.get(fr.frame_idx0 + 1, []), motion_eps=motion_eps)

        global_motion_delta = 0.0
        if features:
            global_motion_delta = math.hypot(motion["global_dx"] - prev_global_dx, motion["global_dy"] - prev_global_dy)

        ff = FrameFeatures(
            frame_idx0=fr.frame_idx0,
            pts_time=fr.pts_time,
            pict_type=fr.pict_type,
            gop_id=fr.gop_id,
            within_gop_idx=fr.within_gop_idx,
            pkt_size=fr.pkt_size,
            pkt_log=pkt_log,
            pkt_jump=pkt_jump,
            mv_count=motion["mv_count"],
            mv_mean=motion["mv_mean"],
            mv_p90=motion["mv_p90"],
            mv_active_ratio=motion["mv_active_ratio"],
            dir_entropy=motion["dir_entropy"],
            global_dx=motion["global_dx"],
            global_dy=motion["global_dy"],
            global_motion_delta=global_motion_delta,
        )
        features.append(ff)
        prev_pkt_log = pkt_log
        prev_global_dx = motion["global_dx"]
        prev_global_dy = motion["global_dy"]

    # Robust normalization over P-frames only
    p_feats = [f for f in features if f.pict_type == "P"]
    if not p_feats:
        return features

    def zmap(attr: str) -> Dict[int, float]:
        vals = [getattr(f, attr) for f in p_feats]
        zs = robust_zscores(vals)
        return {f.frame_idx0: z for f, z in zip(p_feats, zs)}

    z_pkt = zmap("pkt_log")
    z_jump = zmap("pkt_jump")
    z_mv_mean = zmap("mv_mean")
    z_mv_p90 = zmap("mv_p90")
    z_mv_active = zmap("mv_active_ratio")
    z_dir = zmap("dir_entropy")
    z_cam = zmap("global_motion_delta")

    for f in p_feats:
        # Residual proxy is packet size. Motion terms contribute only if available.
        novelty_raw = (
            0.50 * relu(z_pkt.get(f.frame_idx0, 0.0))
            + 0.18 * relu(z_mv_mean.get(f.frame_idx0, 0.0))
            + 0.12 * relu(z_mv_p90.get(f.frame_idx0, 0.0))
            + 0.10 * relu(z_mv_active.get(f.frame_idx0, 0.0))
            + 0.10 * relu(z_dir.get(f.frame_idx0, 0.0))
        )
        boundary_raw = (
            0.55 * relu(z_pkt.get(f.frame_idx0, 0.0))
            + 0.25 * relu(z_jump.get(f.frame_idx0, 0.0))
            + 0.20 * relu(z_cam.get(f.frame_idx0, 0.0))
        )
        f.novelty_raw = float(novelty_raw)
        f.boundary_raw = float(boundary_raw)
        f.debug = {
            "z_pkt": z_pkt.get(f.frame_idx0, 0.0),
            "z_jump": z_jump.get(f.frame_idx0, 0.0),
            "z_mv_mean": z_mv_mean.get(f.frame_idx0, 0.0),
            "z_mv_p90": z_mv_p90.get(f.frame_idx0, 0.0),
            "z_mv_active": z_mv_active.get(f.frame_idx0, 0.0),
            "z_dir": z_dir.get(f.frame_idx0, 0.0),
            "z_cam": z_cam.get(f.frame_idx0, 0.0),
        }

    return features


# -----------------------------
# Selection logic
# -----------------------------


def split_gops(features: List[FrameFeatures]) -> List[List[FrameFeatures]]:
    groups: Dict[int, List[FrameFeatures]] = {}
    for f in features:
        groups.setdefault(f.gop_id, []).append(f)
    return [groups[k] for k in sorted(groups)]


def select_frames_for_video(
    features: List[FrameFeatures],
    boundary_threshold: float,
    acc_threshold: float,
    max_gap: int,
) -> List[FrameFeatures]:
    for f in features:
        f.keep = False
        f.keep_reason = ""
        f.score_accum_at_keep = 0.0

    gops = split_gops(features)
    for gop in gops:
        if not gop:
            continue
        # Always keep first I-frame in each GOP.
        anchor_idx = 0
        gop[0].keep = True
        gop[0].keep_reason = "I-anchor"
        gop[0].score_accum_at_keep = 0.0
        accum = 0.0

        for i in range(1, len(gop)):
            f = gop[i]
            if f.pict_type != "P":
                f.keep = True
                f.keep_reason = "non-P"
                f.score_accum_at_keep = accum
                accum = 0.0
                anchor_idx = i
                continue

            if f.boundary_raw >= boundary_threshold:
                f.keep = True
                f.keep_reason = "boundary"
                f.score_accum_at_keep = accum
                accum = 0.0
                anchor_idx = i
                continue

            accum += max(f.novelty_raw, 0.0)
            gap = i - anchor_idx
            if accum >= acc_threshold:
                f.keep = True
                f.keep_reason = "accum_novelty"
                f.score_accum_at_keep = accum
                accum = 0.0
                anchor_idx = i
                continue

            if gap >= max_gap:
                f.keep = True
                f.keep_reason = "max_gap"
                f.score_accum_at_keep = accum
                accum = 0.0
                anchor_idx = i
                continue

    return features


def mean_kept_per_gop(features: List[FrameFeatures]) -> float:
    gops = split_gops(features)
    vals = [sum(1 for f in g if f.keep) for g in gops if g]
    return float(sum(vals) / max(len(vals), 1))


def auto_boundary_threshold(all_features: List[List[FrameFeatures]], boundary_quantile: float) -> float:
    vals = [f.boundary_raw for video in all_features for f in video if f.pict_type == "P"]
    if not vals:
        return float("inf")
    vals = sorted(vals)
    idx = min(max(int(boundary_quantile * (len(vals) - 1)), 0), len(vals) - 1)
    return float(vals[idx])


def auto_acc_threshold(
    all_features: List[List[FrameFeatures]],
    boundary_threshold: float,
    max_gap: int,
    target_kept_per_gop: float,
) -> float:
    vals = [f.novelty_raw for video in all_features for f in video if f.pict_type == "P"]
    if not vals:
        return float("inf")
    hi = max(sum(sorted(vals, reverse=True)[:10]), 1.0)
    lo = 0.0

    # Binary search on accumulated-threshold to match target kept-per-GOP.
    for _ in range(24):
        mid = (lo + hi) / 2.0
        kept = []
        for video in all_features:
            # work on copies? select mutates keep flags but that's okay per iteration.
            select_frames_for_video(video, boundary_threshold, mid, max_gap)
            kept.append(mean_kept_per_gop(video))
        avg_kept = sum(kept) / max(len(kept), 1)
        if avg_kept > target_kept_per_gop:
            # too many frames kept => threshold should increase
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


# -----------------------------
# Per-video processing
# -----------------------------


def process_one_video(
    video_path: Path,
    input_dir: Path,
    mv_extractor: Optional[str],
    motion_eps: float,
    allow_pktsize_fallback: bool,
    offline_cache_dir: Optional[Path],
) -> Tuple[Path, List[FrameFeatures], Dict[str, Any]]:
    _ = mv_extractor  # kept for backward-compatible function signature

    cache_source = None
    if offline_cache_dir is not None:
        cache_path = cache_json_path_for_video(offline_cache_dir, input_dir, video_path)
        if not cache_path.is_file():
            raise FileNotFoundError(f"offline cache not found: {cache_path}")
        frames, frame_to_mvs, cached_meta = load_offline_video_data(cache_path)
        mv_used = bool(cached_meta.get("motion_extractor_used", False))
        warning = cached_meta.get("warning", None)
        cache_source = str(cache_path)
    else:
        frames = ffprobe_frames(video_path)
        mv_used = False
        frame_to_mvs = None
        warning = None

        try:
            frame_to_mvs = parse_motion_vectors_with_pyav(video_path)
            mv_used = True
        except Exception as exc:
            if not allow_pktsize_fallback:
                raise
            warning = f"PyAV motion-vector read failed; fallback to packet-size-only mode: {exc}"
            frame_to_mvs = None

    features = build_frame_features(frames, frame_to_mvs=frame_to_mvs, motion_eps=motion_eps)

    meta = {
        "video_path": str(video_path),
        "num_frames": len(frames),
        "num_gops": len(split_gops(features)),
        "motion_extractor_used": mv_used,
        "warning": warning,
        "offline_cache_source": cache_source,
    }
    return video_path, features, meta


# -----------------------------
# Writing outputs
# -----------------------------


def save_video_json(out_path: Path, video_path: Path, features: List[FrameFeatures], meta: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_path": str(video_path),
        "meta": meta,
        "frames": [
            {
                "frame_idx0": f.frame_idx0,
                "pts_time": f.pts_time,
                "pict_type": f.pict_type,
                "gop_id": f.gop_id,
                "within_gop_idx": f.within_gop_idx,
                "pkt_size": f.pkt_size,
                "pkt_log": f.pkt_log,
                "pkt_jump": f.pkt_jump,
                "mv_count": f.mv_count,
                "mv_mean": f.mv_mean,
                "mv_p90": f.mv_p90,
                "mv_active_ratio": f.mv_active_ratio,
                "dir_entropy": f.dir_entropy,
                "global_dx": f.global_dx,
                "global_dy": f.global_dy,
                "global_motion_delta": f.global_motion_delta,
                "novelty_raw": f.novelty_raw,
                "boundary_raw": f.boundary_raw,
                "keep": f.keep,
                "keep_reason": f.keep_reason,
                "score_accum_at_keep": f.score_accum_at_keep,
                "debug": f.debug,
            }
            for f in features
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_row(video_path: Path, features: List[FrameFeatures], meta: Dict[str, Any]) -> Dict[str, Any]:
    gops = split_gops(features)
    kept_total = sum(1 for f in features if f.keep)
    p_kept = sum(1 for f in features if f.keep and f.pict_type == "P")
    p_total = sum(1 for f in features if f.pict_type == "P")
    i_total = sum(1 for f in features if f.pict_type == "I")
    kept_per_gop = mean_kept_per_gop(features)
    rows_per_gop = [sum(1 for f in g if f.keep) for g in gops if g]
    return {
        "video_path": str(video_path),
        "num_frames": len(features),
        "num_gops": len(gops),
        "i_total": i_total,
        "p_total": p_total,
        "kept_total": kept_total,
        "kept_p": p_kept,
        "keep_ratio_all": round(kept_total / max(len(features), 1), 6),
        "keep_ratio_p": round(p_kept / max(p_total, 1), 6),
        "mean_kept_per_gop": round(kept_per_gop, 6),
        "min_kept_per_gop": min(rows_per_gop) if rows_per_gop else 0,
        "max_kept_per_gop": max(rows_per_gop) if rows_per_gop else 0,
        "motion_extractor_used": meta.get("motion_extractor_used", False),
        "warning": meta.get("warning", "") or "",
    }


def write_selected_csv(out_path: Path, by_video: List[Tuple[Path, List[FrameFeatures]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_path",
                "frame_idx0",
                "pts_time",
                "pict_type",
                "gop_id",
                "within_gop_idx",
                "keep_reason",
                "novelty_raw",
                "boundary_raw",
                "pkt_size",
                "mv_count",
                "mv_mean",
                "mv_p90",
                "mv_active_ratio",
                "dir_entropy",
            ],
        )
        writer.writeheader()
        for video_path, feats in by_video:
            for ftr in feats:
                if not ftr.keep:
                    continue
                writer.writerow(
                    {
                        "video_path": str(video_path),
                        "frame_idx0": ftr.frame_idx0,
                        "pts_time": f"{ftr.pts_time:.6f}",
                        "pict_type": ftr.pict_type,
                        "gop_id": ftr.gop_id,
                        "within_gop_idx": ftr.within_gop_idx,
                        "keep_reason": ftr.keep_reason,
                        "novelty_raw": f"{ftr.novelty_raw:.6f}",
                        "boundary_raw": f"{ftr.boundary_raw:.6f}",
                        "pkt_size": ftr.pkt_size,
                        "mv_count": ftr.mv_count,
                        "mv_mean": f"{ftr.mv_mean:.6f}",
                        "mv_p90": f"{ftr.mv_p90:.6f}",
                        "mv_active_ratio": f"{ftr.mv_active_ratio:.6f}",
                        "dir_entropy": f"{ftr.dir_entropy:.6f}",
                    }
                )


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Codec-domain frame proposal using ffprobe + PyAV motion vectors")
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--offline_cache_dir",
        type=Path,
        default=None,
        help="If set, read precomputed per-video codec data (.json) from this directory instead of decoding videos online.",
    )
    p.add_argument(
        "--mv_extractor",
        type=str,
        default=None,
        help="Deprecated and ignored. Motion vectors are now read via PyAV.",
    )
    p.add_argument("--workers", type=int, default=max(os.cpu_count() or 4, 4))
    p.add_argument("--motion_eps", type=float, default=0.75, help="Threshold for active local motion magnitude")
    p.add_argument("--allow_pktsize_fallback", action="store_true", help="Allow packet-size-only fallback if PyAV MV reading fails")
    p.add_argument("--boundary_quantile", type=float, default=0.92, help="Quantile used to set the boundary threshold automatically")
    p.add_argument("--target_kept_per_gop", type=float, default=4.0, help="Average kept frames per GOP, including the forced I-frame")
    p.add_argument("--max_gap", type=int, default=4, help="Maximum number of frames from the last kept frame within a GOP")
    p.add_argument("--calib_videos", type=int, default=0, help="Number of videos used for threshold calibration; 0 means all videos")
    p.add_argument("--limit_videos", type=int, default=0, help="Only process the first N videos for quick experiments")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_dir)
    if args.limit_videos and args.limit_videos > 0:
        videos = videos[: args.limit_videos]
    if not videos:
        print(f"No mp4 videos found under: {input_dir}", file=sys.stderr)
        return 1

    use_offline_cache = args.offline_cache_dir is not None
    mv_extractor = "pyav" if av is not None else None
    if (not use_offline_cache) and mv_extractor is None and not args.allow_pktsize_fallback:
        print(
            "PyAV is not available. Please install it with `pip install av`, "
            "or pass --allow_pktsize_fallback to run packet-size-only mode.",
            file=sys.stderr,
        )
        return 2

    if args.mv_extractor:
        print("[WARN] --mv_extractor is deprecated and ignored; using PyAV motion-vector reading instead.")

    print(f"[INFO] Found {len(videos)} videos")
    if use_offline_cache:
        print(f"[INFO] Offline cache dir: {args.offline_cache_dir}")
    else:
        print(f"[INFO] Motion extractor: {mv_extractor or 'NONE (packet-size-only fallback)'}")

    processed: List[Tuple[Path, List[FrameFeatures], Dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                process_one_video,
                video,
                input_dir,
                mv_extractor,
                args.motion_eps,
                args.allow_pktsize_fallback,
                args.offline_cache_dir,
            ): video
            for video in videos
        }
        for fut in as_completed(futs):
            video = futs[fut]
            try:
                processed.append(fut.result())
                print(f"[OK] analyzed {video}")
            except Exception as exc:
                print(f"[ERR] failed {video}: {exc}", file=sys.stderr)

    if not processed:
        print("No videos analyzed successfully.", file=sys.stderr)
        return 3

    processed.sort(key=lambda x: str(x[0]))
    all_features = [feats for _, feats, _ in processed]
    calib_pool = all_features[: args.calib_videos] if args.calib_videos and args.calib_videos > 0 else all_features

    boundary_threshold = auto_boundary_threshold(calib_pool, args.boundary_quantile)
    acc_threshold = auto_acc_threshold(
        calib_pool,
        boundary_threshold=boundary_threshold,
        max_gap=args.max_gap,
        target_kept_per_gop=args.target_kept_per_gop,
    )

    print(f"[INFO] boundary_threshold = {boundary_threshold:.6f}")
    print(f"[INFO] acc_threshold      = {acc_threshold:.6f}")

    summary_rows: List[Dict[str, Any]] = []
    by_video_selected: List[Tuple[Path, List[FrameFeatures]]] = []

    per_video_dir = output_dir / "per_video"
    for video_path, feats, meta in processed:
        select_frames_for_video(feats, boundary_threshold, acc_threshold, args.max_gap)
        summary_rows.append(build_summary_row(video_path, feats, meta))
        by_video_selected.append((video_path, feats))
        rel = video_path.relative_to(input_dir)
        json_path = per_video_dir / rel.with_suffix(".json")
        save_video_json(json_path, video_path, feats, meta)

    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_selected_csv(output_dir / "selected_frames.csv", by_video_selected)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "motion_extractor": mv_extractor,
                "boundary_quantile": args.boundary_quantile,
                "target_kept_per_gop": args.target_kept_per_gop,
                "max_gap": args.max_gap,
                "motion_eps": args.motion_eps,
                "workers": args.workers,
                "allow_pktsize_fallback": args.allow_pktsize_fallback,
                "offline_cache_dir": str(args.offline_cache_dir) if args.offline_cache_dir else None,
                "boundary_threshold": boundary_threshold,
                "acc_threshold": acc_threshold,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Aggregate quick stats.
    avg_keep = sum(r["mean_kept_per_gop"] for r in summary_rows) / max(len(summary_rows), 1)
    avg_ratio = sum(r["keep_ratio_all"] for r in summary_rows) / max(len(summary_rows), 1)
    print(f"[DONE] videos={len(summary_rows)} avg_kept_per_gop={avg_keep:.4f} avg_keep_ratio={avg_ratio:.4f}")
    print(f"[DONE] outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python /mnt/data/codec_frame_sampler.py --input_dir /data/oceanus_ctr/j-shangshouduo-jk/project/data/raw/videomme/data_fps2 --output_dir /data/oceanus_ctr/j-shangshouduo-jk/project/data/raw/videomme/codec_sampling_v1 --allow_pktsize_fallback --workers 8 --target_kept_per_gop 4.0 --max_gap 4 --boundary_quantile 0.92
# python /mnt/data/codec_frame_sampler.py --input_dir /data/oceanus_ctr/j-shangshouduo-jk/project/data/raw/videomme/data_fps2 --output_dir /data/oceanus_ctr/j-shangshouduo-jk/project/data/raw/videomme/codec_sampling_v1 --mv_extractor /path/to/extract_mvs --allow_pktsize_fallback --workers 8 --target_kept_per_gop 4.0 --max_gap 4 --boundary_quantile 0.92

# python /data/oceanus_ctr/j-shangshouduo-jk/myproject/data/code/Video-MME/codec_frame_sampler_without_extract_mvs.py --input_dir /data/oceanus_ctr/j-shangshouduo-jk/project/data/raw/videomme/data_fps2 --output_dir /data/oceanus_ctr/j-shangshouduo-jk/project/data/raw/videomme/codec_sampling_v2_add_action --allow_pktsize_fallback --workers 8 --target_kept_per_gop 4.0 --max_gap 4 --boundary_quantile 0.92
