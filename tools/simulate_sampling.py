#!/usr/bin/env python3
"""Answer "how many frames per second do we need?" without running the model again.

Reads the per-frame counts saved by `run.py --dump-frames` (model run on EVERY
frame = the graders' reference), keeps only some frames, and scores the result
against the full reference with the brief's three thresholds.

Sampling rule (the one the fast run.py will use): split each second into N equal
slots and keep the FIRST frame whose timestamp falls in each slot. This works for
any frame rate (24/25/30/60, variable) because it uses timestamps, not frame numbers.

    python tools/simulate_sampling.py frames/*.frames.jsonl
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import fmt, passes, score  # noqa: E402

CLASS_NAMES = ["person", "car", "dog", "laptop", "cell phone"]


def summarize_second(rows):
    """Same rule as run.py: present in >= half the frames; median people, rounded down."""
    n = len(rows)
    if n == 0:
        return set(), 0
    objects = {CLASS_NAMES[i] for i in range(5) if 2 * sum(1 for r in rows if r[i] > 0) >= n}
    p = sorted(r[0] for r in rows)
    people = p[n // 2] if n % 2 else (p[n // 2 - 1] + p[n // 2]) // 2
    return objects, people


def per_second(video, frames, keep):
    """frames: [[t, p, car, dog, laptop, phone], ...]. keep(t, sec) -> bool."""
    by_sec = defaultdict(list)
    last = -1
    for f in frames:
        t = f[0]
        s = math.floor(t + 1e-6)
        last = max(last, s)
        if keep(t, s):
            by_sec[s].append(f[1:])
    return {(video, s): summarize_second(by_sec.get(s, [])) for s in range(last + 1)}


def slot_sampler(n_per_sec):
    filled = set()

    def keep(t, s):
        slot = min(int((t - s) * n_per_sec), n_per_sec - 1)
        if (s, slot) in filled:
            return False
        filled.add((s, slot))
        return True
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", help="*.frames.jsonl files from run.py --dump-frames")
    ap.add_argument("--rates", default="1,2,3,4,5,6,8,10,15")
    args = ap.parse_args()

    videos = []
    for path in args.dumps:
        with open(path) as f:
            videos += [json.loads(line) for line in f]
    n_frames = sum(len(v["frames"]) for v in videos)

    ref = {}
    for v in videos:
        ref.update(per_second(v["video"], v["frames"], lambda t, s: True))
    print(f"{len(videos)} videos, {len(ref)} seconds, {n_frames} frames "
          f"(avg {n_frames / max(len(ref), 1):.1f} fps)\n")

    print(f"{'per sec':>7} {'frames used':>12}   result")
    for n in [int(x) for x in args.rates.split(",")]:
        pred, used = {}, 0
        for v in videos:
            keep = slot_sampler(n)
            counted = []

            def k(t, s, keep=keep):
                ok = keep(t, s)
                counted.append(ok)
                return ok
            pred.update(per_second(v["video"], v["frames"], k))
            used += sum(counted)
        s = score(ref, pred)
        print(f"{n:>7} {used / n_frames * 100:>11.1f}%   {fmt(s)}")

    # worst videos at a middle rate, to see where errors come from
    print("\nper-video at 5 per sec (worst 5 by exact):")
    rows = []
    for v in videos:
        r = per_second(v["video"], v["frames"], lambda t, s: True)
        p = per_second(v["video"], v["frames"], slot_sampler(5))
        rows.append((score(r, p), v["video"]))
    for s, name in sorted(rows, key=lambda x: x[0]["exact"])[:5]:
        print(f"  {name}: {fmt(s)}")


if __name__ == "__main__":
    main()
