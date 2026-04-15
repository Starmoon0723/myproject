
import os
import re
import json
import ast
import string
import pandas as pd

from lmms_eval.tasks.mmvu.utils import evaluate_with_llm_judge


# ---------------------------
# Final-answer extraction
# ---------------------------

_FINAL_MC_PATTERNS = [
    r"(?:therefore,?\s*)?the\s*final\s*answer\s*is\s*[:：]?\s*([A-E])\b",
    r"\bfinal\s*answer\s*[:：]?\s*([A-E])\b",
    r"\banswer\s*[:：]?\s*([A-E])\b",
    r"\boption\s*([A-E])\b",
]

_FINAL_OE_PATTERNS = [
    r"(?:therefore,?\s*)?the\s*final\s*answer\s*is\s*[:：]?\s*(.+)$",
    r"\bfinal\s*answer\s*[:：]?\s*(.+)$",
    r"\banswer\s*[:：]?\s*(.+)$",
]


def _clean_text(s: str) -> str:
    s = str(s).strip()
    # Remove wrapping quotes
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        s = s[1:-1].strip()
    # Trim trailing punctuation
    s = s.strip().strip(string.punctuation).strip()
    return s


def extract_final_answer_multiple_choice(pred: str, choices: dict | None = None) -> str | None:
    """
    Return a single letter 'A'..'E' if confidently extracted, else None.
    """
    s = str(pred or "").strip()

    # If the model explicitly says "none of the above", treat as no valid option.
    if re.search(r"\bnone\s+of\s+the\s+above\b", s, flags=re.IGNORECASE):
        return None

    # Prefer explicit "final answer is: X" style patterns (take the LAST match).
    for pat in _FINAL_MC_PATTERNS:
        ms = list(re.finditer(pat, s, flags=re.IGNORECASE | re.MULTILINE))
        if ms:
            return ms[-1].group(1).upper()

    # If it outputs choice text instead of letter, try to map to a letter.
    # Example: "Low-pass filter" for an MC question where one option text matches.
    if isinstance(choices, dict) and choices:
        pred_clean = " ".join(_clean_text(s).lower().split())
        best_letter = None
        best_len = 0
        for letter, txt in choices.items():
            if not txt:
                continue
            opt_clean = " ".join(_clean_text(txt).lower().split())
            # If prediction contains option text (or vice versa), pick the longer match.
            if opt_clean and (opt_clean in pred_clean or pred_clean in opt_clean):
                if len(opt_clean) > best_len:
                    best_len = len(opt_clean)
                    best_letter = str(letter).upper()
        if best_letter in {"A", "B", "C", "D", "E"}:
            return best_letter

    # Fallback: take the LAST standalone A-E in the entire output (still risky, but last is better than first).
    letters = re.findall(r"\b([A-E])\b", s, flags=re.IGNORECASE)
    return letters[-1].upper() if letters else None


def extract_final_answer_open_ended(pred: str) -> str:
    """
    Return a short final answer string extracted from prediction.
    """
    s = str(pred or "").strip()

    # Prefer explicit "final answer is: ..." patterns (take the LAST match).
    for pat in _FINAL_OE_PATTERNS:
        ms = list(re.finditer(pat, s, flags=re.IGNORECASE | re.MULTILINE))
        if ms:
            return _clean_text(ms[-1].group(1))

    # Fallback: last non-empty line
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        return _clean_text(lines[-1])

    return _clean_text(s)


def extract_final_answer(doc: dict, pred: str) -> str:
    """
    Extract final answer and return a compact string to feed into lmms_eval's evaluator.
    - multiple-choice: return a single letter if possible, otherwise return empty string
    - open-ended: return extracted short answer
    """
    qtype = doc.get("question_type", "")
    if qtype == "multiple-choice":
        letter = extract_final_answer_multiple_choice(pred, choices=doc.get("choices"))
        return letter or ""
    else:
        return extract_final_answer_open_ended(pred)


# ---------------------------
# Your evaluation script (modified)
# ---------------------------

def main():
    # result_path = "/data/oceanus_share/shangshouduo-jk/project/results/refine/Qwen3-VL-8B-Instruct/MMVU/T20260121_G85738573/Qwen3-VL-8B-Instruct_MMVU.jsonl"
    result_path = "/data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/aks/clip/Qwen3-VL-8B-Instruct/MMVU/T20260203_G85738573/Qwen3-VL-8B-Instruct_MMVU.jsonl"   # 59.2
    # result_path = "/data/oceanus_share/shangshouduo-jk/myproject/output/results/baselines/q-frame/longclip/Qwen3-VL-8B-Instruct/MMVU/T20260204_G85738573/Qwen3-VL-8B-Instruct_MMVU.jsonl" # 53.7
    output_dir = os.path.dirname(result_path)
    output_jsonl = os.path.join(output_dir, "evaluation_results_only_true_new.jsonl")

    _, ext = os.path.splitext(result_path)
    ext = ext.lower()

    if ext == ".xlsx":
        df = pd.read_excel(result_path)
    elif ext == ".jsonl":
        df = pd.read_json(result_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Please provide .xlsx or .jsonl")

    if "index" not in df.columns:
        df = df.reset_index()  # adds 0-based index column named "index"

    # Optional but recommended: avoid duplicated samples
    # If your "index" is unique sample id, uncomment:
    # df = df.drop_duplicates(subset=["index"], keep="first").reset_index(drop=True)

    results = []
    output_lines = []

    for _, row in df.iterrows():
        # Safe parse choices
        choices = row.get("choices", None)

        if pd.isna(choices):
            choices = None
        elif isinstance(choices, str):
            # JSONL sometimes stores dict as JSON string; excel sometimes stores python dict repr
            s = choices.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    choices = json.loads(s)
                except Exception:
                    try:
                        choices = ast.literal_eval(s)
                    except Exception:
                        choices = None
            else:
                try:
                    choices = ast.literal_eval(s)
                except Exception:
                    choices = None
        elif isinstance(choices, dict):
            pass
        else:
            choices = None

        doc = {
            "question_type": row["question_type"],
            "answer": row["answer"],
            "choices": choices,
        }

        raw_pred = row.get("prediction", "")
        # Key change: extract final answer first
        final_pred = extract_final_answer(doc, raw_pred)

        # Now call lmms_eval evaluator on the *final* answer only
        is_correct, method = evaluate_with_llm_judge(doc, final_pred)

        results.append(bool(is_correct))

        out_entry = {
            "index": int(row["index"]),
            "answer": row["answer"],
            "prediction_raw": str(raw_pred),
            "prediction_final": str(final_pred),
            "is_correct": bool(is_correct),
            "eval_method": method,
        }

        if is_correct:
            output_lines.append(out_entry)

    acc = sum(results) / len(results) if results else 0.0
    print(f"Accuracy: {acc:.6f}  (N={len(results)})")

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for entry in output_lines:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Evaluation results (only correct) saved to: {output_jsonl}")


if __name__ == "__main__":
    main()
