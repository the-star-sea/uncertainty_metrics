#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute AUARC (0..20% rejection) as mean conditional accuracy from harness JSON.
Understands either:
  - a top-level {"samples": {...task_name: [example, ...], ...}} results file, or
  - a plain list/{"samples": [...]} predictions file.
"""
import argparse, json
import numpy as np

def load_samples(path, task=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case A: consolidated results dict with "samples": {task_name: [examples]}
    if isinstance(data, dict) and "samples" in data and isinstance(data["samples"], dict):
        if task is not None:
            return data["samples"].get(task, [])
        # flatten across tasks if none specified
        out = []
        for _, lst in data["samples"].items():
            out.extend(lst)
        return out

    # Case B: {"samples":[...]} or a raw list
    if isinstance(data, dict) and "samples" in data and isinstance(data["samples"], list):
        return data["samples"]
    if isinstance(data, list):
        return data

    raise ValueError("Unrecognized predictions format")

def _first_or_none(lst):
    return lst[0] if isinstance(lst, list) and len(lst) > 0 else None

def iter_predictions_with_scores(examples):
    """
    Yield tuples (pred_text, score_dict, target_text, correct_flag_guess)
    from harness examples that include ["resps"] and ["scores"] lists.
    """
    for ex in examples:
        # typical shape: resps: [[text]] ; scores: [[{...}]]
        pred_text = None
        score = None
        target = ex.get("target", None)

        resps = ex.get("resps", None)
        if isinstance(resps, list) and len(resps) > 0:
            pred_text = _first_or_none(resps[0])

        scores = ex.get("scores", None)
        if isinstance(scores, list) and len(scores) > 0:
            score = _first_or_none(scores[0])

        # fallback correctness by string equality
        correct = None
        if pred_text is not None and target is not None:
            correct = (str(pred_text).strip() == str(target).strip())

        yield pred_text, score, target, correct

def get_conf(score, field="geo_prob"):
    if not isinstance(score, dict):
        return float("nan")
    if field in score and score[field] is not None:
        return float(score[field])
    # fallback: exp(mean_logprob)
    ml = score.get("mean_logprob", None)
    return float(np.exp(ml)) if ml is not None else float("nan")

def compute_auarc(examples, field="geo_prob", max_rej=0.20, step=0.01):
    rows = []
    for pred, score, target, correct in iter_predictions_with_scores(examples):
        conf = get_conf(score, field)
        if conf == conf:  # not NaN
            if correct is None:
                # if task didn't include "correct", fallback to text equality already done above
                correct = (str(pred).strip() == str(target).strip()) if (pred is not None and target is not None) else False
            rows.append((conf, 1 if correct else 0))
    if not rows:
        return float("nan"), []

    rows.sort(key=lambda x: -x[0])
    corr = np.array([r[1] for r in rows], dtype=np.int32)
    N = len(rows)
    cumsum = np.cumsum(corr)

    rejs = np.arange(0.0, max_rej + 1e-9, step)
    accs = []
    for r in rejs:
        keep = int(round((1.0 - r) * N))
        keep = max(1, min(N, keep))
        accs.append(cumsum[keep - 1] / keep)
    auarc = float(np.mean(accs))
    return auarc, list(zip(rejs.tolist(), accs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True)
    ap.add_argument("--task", default=None, help="If consolidated results, choose a specific task to evaluate")
    ap.add_argument("--score_field", default="geo_prob")
    ap.add_argument("--max_reject", type=float, default=0.20)
    ap.add_argument("--step", type=float, default=0.01)
    args = ap.parse_args()

    examples = load_samples(args.pred_json, task=args.task)
    auarc, curve = compute_auarc(examples, field=args.score_field, max_rej=args.max_reject, step=args.step)

    print(f"AUARC (mean conditional accuracy, 0..{int(args.max_reject*100)}% rejection): {auarc:.4f}")
    for r, a in curve:
        print(f"reject={r:.2%}\tacc={a:.4f}")

if __name__ == "__main__":
    main()
