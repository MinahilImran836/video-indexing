#!/usr/bin/env python3
"""Score a results file against a reference, using the brief's three thresholds.

    python tools/evaluate.py --ref reference.jsonl --pred results.jsonl

Both files may be .jsonl or .jsonl.gz. Only videos present in the reference are
scored. Thresholds: objects F1 >= 0.95, people exact >= 85%, people within 1 >= 95%.
"""
import argparse
import gzip
import json

F1_MIN, EXACT_MIN, WITHIN1_MIN = 0.95, 0.85, 0.95


def load(path):
    """-> {(video, second): (set(objects), people)}, set of videos with an error line"""
    opener = gzip.open if str(path).endswith(".gz") else open
    secs, errors = {}, set()
    with opener(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if "error" in d:
                errors.add(d["video"])
            else:
                secs[(d["video"], d["second"])] = (set(d["objects"]), d["people"])
    return secs, errors


def score(ref, pred):
    """ref, pred: {(video, second): (set(objects), people)}. Missing seconds count as wrong."""
    tp = fp = fn = exact = within1 = 0
    for key, (r_obj, r_ppl) in ref.items():
        p_obj, p_ppl = pred.get(key, (set(), None))
        tp += len(r_obj & p_obj)
        fp += len(p_obj - r_obj)
        fn += len(r_obj - p_obj)
        if p_ppl is not None:
            exact += p_ppl == r_ppl
            within1 += abs(p_ppl - r_ppl) <= 1
    # extra seconds in pred that the reference doesn't have
    extra = [k for k in pred if k not in ref and k[0] in {v for v, _ in ref}]
    for k in extra:
        fp += len(pred[k][0])
    n = len(ref)
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
    return {
        "seconds": n,
        "extra_seconds": len(extra),
        "f1": f1,
        "exact": exact / n if n else 1.0,
        "within1": within1 / n if n else 1.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def passes(s):
    return s["f1"] >= F1_MIN and s["exact"] >= EXACT_MIN and s["within1"] >= WITHIN1_MIN


def fmt(s):
    return (f"F1 {s['f1']:.4f}  exact {s['exact'] * 100:.2f}%  within1 {s['within1'] * 100:.2f}%  "
            f"({s['seconds']} s, tp {s['tp']} fp {s['fp']} fn {s['fn']})  "
            f"{'PASS' if passes(s) else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()
    ref, ref_err = load(args.ref)
    pred, pred_err = load(args.pred)
    videos = {v for v, _ in ref}
    pred = {k: v for k, v in pred.items() if k[0] in videos}
    print(fmt(score(ref, pred)))
    if ref_err or pred_err:
        print(f"error lines: ref {len(ref_err)}, pred {len(pred_err)}; "
              f"only in pred: {sorted(pred_err - ref_err)[:10]}")


if __name__ == "__main__":
    main()
