#!/usr/bin/env python3
"""BASELINE (frozen copy of run.py v1, tag baseline-v1) - every frame, used as our reference.

Video indexing with YOLO11l.

For every second of every video in a folder of FineVideo Parquet files, record
which of five objects are present and how many people are visible.

    python run.py --input <folder of .parquet files> --output results.jsonl

VERSION 1 (baseline): runs the model on EVERY frame, exactly like the graders'
reference. Slow, but it is our ground truth for checking faster versions.
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import pyarrow.parquet as pq

# COCO class id -> name, in the order we store per-frame counts.
CLASS_IDS = [0, 2, 16, 63, 67]
CLASS_NAMES = ["person", "car", "dog", "laptop", "cell phone"]

IMGSZ = 640
CONF = 0.4
BATCH = 16  # frames sent to the GPU at once


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(weights):
    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(weights)
    return model, device


def detect(model, device, frames):
    """Run YOLO on a list of BGR frames. Returns one 5-int count list per frame."""
    results = model.predict(
        frames,
        imgsz=IMGSZ,
        conf=CONF,
        classes=CLASS_IDS,   # other classes are dropped; NMS is class-aware so
        device=device,       # this does not change results for our 5 classes
        verbose=False,
    )
    out = []
    for r in results:
        cls = r.boxes.cls.int().tolist()
        out.append([cls.count(c) for c in CLASS_IDS])
    return out


# --------------------------------------------------------------------------- #
# Per-second summary (section 5 of the brief)
# --------------------------------------------------------------------------- #
def summarize_second(rows):
    """rows: list of 5-int count lists, one per frame in this second."""
    n = len(rows)
    if n == 0:
        return [], 0
    # present if found in at least half of the frames
    objects = [
        CLASS_NAMES[i]
        for i in range(len(CLASS_IDS))
        if 2 * sum(1 for r in rows if r[i] > 0) >= n
    ]
    # median people count, rounded down when halfway between two numbers
    p = sorted(r[0] for r in rows)
    people = p[n // 2] if n % 2 else (p[n // 2 - 1] + p[n // 2]) // 2
    return objects, people


def second_lines(video_id, per_sec, upto):
    """Output lines for seconds 0..upto (inclusive)."""
    lines = []
    for s in range(upto + 1):
        objects, people = summarize_second(per_sec.get(s, []))
        lines.append({"video": video_id, "second": s, "objects": objects, "people": people})
    return lines


# --------------------------------------------------------------------------- #
# One video
# --------------------------------------------------------------------------- #
def process_video(video_id, mp4_bytes, model, device, tmpdir, dump):
    """Returns the list of output lines for this video (incl. an error line if any)."""
    if not mp4_bytes:
        return [{"video": video_id, "error": "empty video bytes"}]

    path = os.path.join(tmpdir, "video.mp4")
    with open(path, "wb") as f:
        f.write(mp4_bytes)

    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return [{"video": video_id, "error": "could not open video"}]
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # apply rotation tag

    per_sec = {}       # second -> list of per-frame counts
    frame_log = []     # [t, counts...] per frame, for --dump-frames
    batch, batch_secs, batch_ts = [], [], []
    t0 = None
    last_sec = -1

    def flush():
        counts = detect(model, device, batch)
        for s, t, c in zip(batch_secs, batch_ts, counts):
            per_sec.setdefault(s, []).append(c)
            if dump is not None:
                frame_log.append([round(t, 4)] + c)
        batch.clear(); batch_secs.clear(); batch_ts.clear()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t0 is None:
                t0 = t
            rel = t - t0
            sec = math.floor(rel + 1e-6)
            batch.append(frame); batch_secs.append(sec); batch_ts.append(rel)
            last_sec = max(last_sec, sec)
            if len(batch) >= BATCH:
                flush()
        if batch:
            flush()
    except Exception as e:  # failed partway: keep finished seconds, add error
        cap.release()
        lines = second_lines(video_id, per_sec, last_sec - 1) if last_sec > 0 else []
        lines.append({"video": video_id, "error": f"failed partway: {type(e).__name__}"})
        return lines
    cap.release()

    if last_sec < 0:
        return [{"video": video_id, "error": "could not decode video"}]

    if dump is not None:
        dump.write(json.dumps({"video": video_id, "frames": frame_log}) + "\n")
    return second_lines(video_id, per_sec, last_sec)


# --------------------------------------------------------------------------- #
# One Parquet file
# --------------------------------------------------------------------------- #
def process_parquet(path, model, device, out, tmpdir, dump, max_videos, rows=None):
    name = path.name
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        out.write(json.dumps({"video": f"{name}:0", "error": f"could not read parquet: {type(e).__name__}"}) + "\n")
        return

    row = 0
    for rg in range(pf.num_row_groups):
        n_rows = pf.metadata.row_group(rg).num_rows
        try:
            col = pf.read_row_group(rg, columns=["mp4"]).column("mp4")
        except Exception:
            for i in range(n_rows):  # whole row group unreadable
                out.write(json.dumps({"video": f"{name}:{row + i}", "error": "could not read row"}) + "\n")
            row += n_rows
            continue

        for i in range(n_rows):
            if max_videos is not None and row >= max_videos:
                return
            if rows is not None and row not in rows:
                row += 1
                continue
            video_id = f"{name}:{row}"
            start = time.time()
            try:
                lines = process_video(video_id, col[i].as_py(), model, device, tmpdir, dump)
            except Exception as e:
                lines = [{"video": video_id, "error": f"{type(e).__name__}"}]
            for line in lines:
                out.write(json.dumps(line) + "\n")
            out.flush()
            log(f"{video_id}: {len(lines)} lines in {time.time() - start:.1f}s")
            row += 1
        del col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of .parquet files")
    ap.add_argument("--output", required=True, help="output .jsonl file")
    ap.add_argument("--weights", default=None, help="path to yolo11l.pt")
    # development options (not used by the graders)
    ap.add_argument("--dump-frames", default=None, help="also write per-frame counts here")
    ap.add_argument("--max-videos", type=int, default=None, help="only the first N videos of each file")
    ap.add_argument("--rows", default=None, help="only these row numbers, e.g. 3,17,25")
    args = ap.parse_args()

    weights = args.weights
    if weights is None:
        local = Path(__file__).resolve().parent / "yolo11l.pt"
        weights = str(local) if local.exists() else "yolo11l.pt"

    files = sorted(Path(args.input).glob("*.parquet"))
    log(f"{len(files)} parquet files")
    model, device = load_model(weights)

    tmp_root = "/dev/shm" if os.path.isdir("/dev/shm") else None
    start = time.time()
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmpdir, \
            open(args.output, "w") as out:
        dump = open(args.dump_frames, "w") if args.dump_frames else None
        for path in files:
            log(f"== {path.name}")
            rows = {int(x) for x in args.rows.split(",")} if args.rows else None
            process_parquet(path, model, device, out, tmpdir, dump, args.max_videos, rows)
        if dump:
            dump.close()
    log(f"done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
