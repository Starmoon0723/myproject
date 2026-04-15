#!/usr/bin/env python3
"""Convert Video-MME parquet split to a TSV file.

Output schema:
index\tvideo_path\tquestion\tcandidates\tanswer
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def pick_column(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find column for {label}. Available columns: {list(df.columns)}")


def normalize_candidates(value: Any) -> str:
    if isinstance(value, list):
        return str(value)
    if isinstance(value, tuple):
        return str(list(value))
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return str(list(parsed))
            except Exception:
                pass
        return str(value)
    return str(value)


def normalize_answer(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text.upper() in {"A", "B", "C", "D"}:
            return text.upper()
        if text.isdigit():
            idx = int(text)
            if 0 <= idx <= 3:
                return "ABCD"[idx]
            if 1 <= idx <= 4:
                return "ABCD"[idx - 1]
        return text
    if isinstance(value, (int, float)):
        idx = int(value)
        if 0 <= idx <= 3:
            return "ABCD"[idx]
        if 1 <= idx <= 4:
            return "ABCD"[idx - 1]
    return str(value)


def is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(256)
        return b"version https://git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def load_input_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if is_git_lfs_pointer(path):
        raise ValueError(
            f"{path} looks like a Git LFS pointer file, not the real dataset file. "
            "Please run `git lfs pull` or re-download the actual data file."
        )

    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise ValueError(
                f"Failed to read parquet file: {path}\n"
                f"Original error: {e}\n"
                "This usually means the file is corrupted, incomplete, or not actually parquet."
            ) from e
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=(suffix == ".jsonl"))
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")

    raise ValueError(
        f"Unsupported input format: {path.suffix}. "
        "Supported: .parquet, .json, .jsonl, .csv, .tsv"
    )


def convert(input_path: Path, tsv_path: Path) -> None:
    df = load_input_table(input_path)

    video_col = pick_column(df, ["video_path", "video", "video_id", "videoID", "video_name"], "video_path")
    question_col = pick_column(df, ["question"], "question")
    candidates_col = pick_column(df, ["candidates", "options", "choices"], "candidates")
    answer_col = pick_column(df, ["answer", "correct_answer", "answer_idx", "answer_id"], "answer")

    out = pd.DataFrame(
        {
            "index": range(len(df)),
            "video_path": df[video_col].astype(str),
            "question": df[question_col].astype(str),
            "candidates": df[candidates_col].map(normalize_candidates),
            "answer": df[answer_col].map(normalize_answer),
        }
    )

    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(tsv_path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Video-MME parquet to TSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("videomme/test-00000-of-00001.parquet"),
        help="Input file path (.parquet/.json/.jsonl/.csv/.tsv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("videomme/video-mme.tsv"),
        help="Output TSV path.",
    )
    args = parser.parse_args()

    convert(args.input, args.output)
    print(f"Saved TSV to: {args.output}")


if __name__ == "__main__":
    main()
