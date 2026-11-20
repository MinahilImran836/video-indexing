#!/usr/bin/env python3
"""Frame-by-frame check: does the fast version see the same thing as the baseline
on the frames it keeps?

    python tools/compare_frames.py --ref baseline.frames.jsonl --pred fast.frames.jsonl

Both files come from --dump-frames. Frames are matched by timestamp. With FP32 the
counts should be identical (same decoder, same preprocessing, same model); with FP16
a small number of frames may differ by one detection near the 0.4 threshold.
"""
import argparse
import json
from collections import Counter

NAMES = ["person", "car", "dog", "laptop", "cell phone"]


def load(paths):
    out = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                out[d["video"]] = {round(fr[0], 3): fr[1:] for fr in d["frames"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--pred", nargs="+", required=True)
    args = ap.parse_args()
    ref, pred = load(args.ref), load(args.pred)

    matched = same = missing = 0
    diff_by_class = Counter()
    worst = Counter()
    for vid, frames in pred.items():
        if vid not in ref:
            continue
        for t, counts in frames.items():
            r = ref[vid].get(t)
            if r is None:
                missing += 1
                continue
            matched += 1
            if r == counts:
                same += 1
            else:
                worst[vid] += 1
                for i, (a, b) in enumerate(zip(r, counts)):
                    if a != b:
                        diff_by_class[NAMES[i]] += 1
    print(f"videos compared: {len(set(pred) & set(ref))}")
    print(f"frames matched by timestamp: {matched} (not found in ref: {missing})")
    print(f"identical counts: {same}/{matched} = {same / max(matched, 1) * 100:.2f}%")
    if diff_by_class:
        print("frames with a different count, by class:", dict(diff_by_class))
        print("videos with most differing frames:", worst.most_common(5))


if __name__ == "__main__":
    main()
