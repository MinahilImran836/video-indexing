#!/usr/bin/env python3
"""Video indexing with YOLO11l - fast version.

For every second of every video in a folder of FineVideo Parquet files, record
which of five objects are present and how many people are visible.

    python run.py --input <folder of .parquet files> --output results.jsonl

How it is fast while staying correct:
  * Decoder processes (CPU) decode several videos in parallel. Every frame is
    decoded (needed for timestamps and because video frames depend on each
    other), but only SAMPLE_FPS frames per second are converted to images:
    each second is split into SAMPLE_FPS equal slots and the first frame in
    each slot is kept (rule validated by tools/simulate_sampling.py).
  * Kept frames are letterboxed in the decoder processes (exact copy of the
    Ultralytics LetterBox) and sent to the main process in chunks.
  * The main process runs YOLO11l in FP16 on the GPU on whole chunks, then
    applies the Ultralytics NMS with the brief's settings.
  * Per second: object present if found in >= half of the kept frames; people
    = median count (rounded down on a tie).
The model settings are the brief's: yolo11l.pt, 640, conf 0.4, class-aware NMS
IoU 0.7, BGR frames, rotation tag applied (OpenCV auto-orientation).
"""
import os

# We never plot. Ultralytics imports matplotlib, which crashes if MPLBACKEND points to a
# backend that isn't installed (e.g. Jupyter's inline backend leaking into a subprocess).
os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
import math
import multiprocessing as mp
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

CLASS_IDS = [0, 2, 16, 63, 67]
CLASS_NAMES = ["person", "car", "dog", "laptop", "cell phone"]
IMGSZ = 640
STRIDE = 32
CONF = 0.4
IOU = 0.7
MAX_DET = 300

SAMPLE_FPS = 5      # frames kept per second of video
CHUNK = 32          # frames per message from decoder to GPU (= GPU batch size)
STALL_S = 600       # a decoder silent this long on one video is treated as hung


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Preprocessing - exact copy of ultralytics LetterBox(640, auto=True, stride=32)
# (what model.predict() uses when all images in a batch have the same shape)
# --------------------------------------------------------------------------- #
def letterbox(img):
    h, w = img.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = ((IMGSZ - new_unpad[0]) % STRIDE) / 2
    dh = ((IMGSZ - new_unpad[1]) % STRIDE) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                              value=(114, 114, 114))


# --------------------------------------------------------------------------- #
# Decoder process
# --------------------------------------------------------------------------- #
def decoder_main(wid, task_q, out_q, sample_fps, n_threads, tmp_dir):
    cv2.setNumThreads(1)
    path = os.path.join(tmp_dir, f"video_{wid}_{os.getpid()}.mp4")
    try:
        while True:
            task = task_q.get()
            if task is None:
                return
            vid, data = task
            out_q.put(("start", wid, vid))
            decode_video(wid, vid, data, path, out_q, sample_fps, n_threads)
    finally:
        if os.path.exists(path):
            os.remove(path)


def decode_video(wid, vid, data, path, out_q, sample_fps, n_threads):
    if not data:
        out_q.put(("error", wid, vid, "empty video bytes", -1))
        return
    with open(path, "wb") as f:
        f.write(data)

    params = []
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        params = [cv2.CAP_PROP_N_THREADS, n_threads]
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, params)
    if not cap.isOpened():
        out_q.put(("error", wid, vid, "could not open video", -1))
        return
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # apply rotation tag

    buf, secs, ts = [], [], []
    filled = set()
    st = {"decode": 0.0, "convert": 0.0, "letterbox": 0.0, "send": 0.0, "decoded": 0, "kept": 0}
    clock = time.perf_counter
    t0 = None
    last_sec = -1

    def flush():
        if buf:
            c = clock()
            out_q.put(("frames", wid, vid, np.stack(buf), list(secs), list(ts)))
            st["send"] += clock() - c   # large = GPU side is the bottleneck (queue full)
            buf.clear(); secs.clear(); ts.clear()

    try:
        while True:
            c = clock()
            ok = cap.grab()             # decode only, no colour conversion
            st["decode"] += clock() - c
            if not ok:
                break
            st["decoded"] += 1
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t0 is None:
                t0 = t
            rel = t - t0
            sec = math.floor(rel + 1e-6)
            last_sec = max(last_sec, sec)
            slot = min(int((rel - sec) * sample_fps), sample_fps - 1)
            if (sec, slot) in filled:
                continue
            filled.add((sec, slot))
            c = clock()
            ok, frame = cap.retrieve()  # BGR, rotated
            st["convert"] += clock() - c
            if not ok:
                break
            c = clock()
            img = letterbox(frame)
            st["letterbox"] += clock() - c
            st["kept"] += 1
            if buf and img.shape != buf[0].shape:  # resolution changed mid-video
                flush()
            buf.append(img); secs.append(sec); ts.append(rel)
            if len(buf) >= CHUNK:
                flush()
        flush()
    except Exception as e:
        flush()
        out_q.put(("stats", wid, vid, st))
        out_q.put(("error", wid, vid, f"failed partway: {type(e).__name__}", last_sec))
        return
    finally:
        cap.release()
    out_q.put(("stats", wid, vid, st))

    if last_sec < 0:
        out_q.put(("error", wid, vid, "could not decode video", -1))
    else:
        out_q.put(("done", wid, vid, last_sec))


# --------------------------------------------------------------------------- #
# Parquet reader thread (main process): feeds video bytes to the decoders
# --------------------------------------------------------------------------- #
def reader_main(files, task_q, out_q, submitted, n_workers, max_videos, rows):
    for path in files:
        name = path.name
        try:
            pf = pq.ParquetFile(path)
        except Exception as e:
            out_q.put(("lines", [{"video": f"{name}:0",
                                  "error": f"could not read parquet: {type(e).__name__}"}]))
            continue
        row = 0
        for rg in range(pf.num_row_groups):
            n_rows = pf.metadata.row_group(rg).num_rows
            try:
                col = pf.read_row_group(rg, columns=["mp4"]).column("mp4")
            except Exception:
                out_q.put(("lines", [{"video": f"{name}:{row + i}", "error": "could not read row"}
                                     for i in range(n_rows)]))
                row += n_rows
                continue
            for i in range(n_rows):
                if max_videos is not None and row >= max_videos:
                    break
                if rows is None or row in rows:
                    vid = f"{name}:{row}"
                    try:
                        data = col[i].as_py()
                    except Exception:
                        data = None
                    submitted.add(vid)
                    task_q.put((vid, data))
                row += 1
            del col
    for _ in range(n_workers):
        task_q.put(None)
    out_q.put(("reader_done",))


# --------------------------------------------------------------------------- #
# GPU detector
# --------------------------------------------------------------------------- #
class Detector:
    def __init__(self, weights, half):
        import torch
        from ultralytics.nn.autobackend import AutoBackend
        from ultralytics.utils import ops

        self.torch, self.ops = torch, ops
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoBackend(weights=weights, device=device, fp16=half, fuse=True, verbose=False)
        self.model.eval()
        self.device = device
        self.half = self.model.fp16
        self.classes = torch.tensor(CLASS_IDS, device=device)
        log(f"model on {device}, fp16={self.half}")

    def __call__(self, batch):
        """batch: uint8 (B, H, W, 3) BGR letterboxed -> list of 5-int counts per frame."""
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(batch).to(self.device, non_blocking=True)
            x = x.flip(-1).permute(0, 3, 1, 2).contiguous()   # BGR->RGB, BHWC->BCHW
            x = x.half() if self.half else x.float()
            x /= 255
            preds = self.model(x)
            dets = self.ops.non_max_suppression(preds, CONF, IOU, classes=CLASS_IDS,
                                                agnostic=False, max_det=MAX_DET)
            counts = torch.stack([(d[:, 5:6].int() == self.classes).sum(0) for d in dets])
            return counts.tolist()


# --------------------------------------------------------------------------- #
# Per-second summary (section 5 of the brief)
# --------------------------------------------------------------------------- #
def summarize_second(rows):
    n = len(rows)
    if n == 0:
        return [], 0
    objects = [CLASS_NAMES[i] for i in range(len(CLASS_IDS))
               if 2 * sum(1 for r in rows if r[i] > 0) >= n]
    p = sorted(r[0] for r in rows)
    people = p[n // 2] if n % 2 else (p[n // 2 - 1] + p[n // 2]) // 2
    return objects, people


def second_lines(vid, per_sec, upto):
    lines = []
    for s in range(upto + 1):
        objects, people = summarize_second(per_sec.get(s, []))
        lines.append({"video": vid, "second": s, "objects": objects, "people": people})
    return lines


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of .parquet files")
    ap.add_argument("--output", required=True, help="output .jsonl file")
    ap.add_argument("--weights", default=None, help="path to yolo11l.pt")
    ap.add_argument("--workers", type=int, default=None, help="decoder processes")
    ap.add_argument("--sample-fps", type=int, default=SAMPLE_FPS)
    ap.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    # development options (not used by the graders)
    ap.add_argument("--dump-frames", default=None, help="also write per-frame counts here")
    ap.add_argument("--max-videos", type=int, default=None, help="only the first N videos of each file")
    ap.add_argument("--rows", default=None, help="only these row numbers, e.g. 3,17,25")
    args = ap.parse_args()

    weights = args.weights
    if weights is None:
        local = Path(__file__).resolve().parent / "yolo11l.pt"
        weights = str(local) if local.exists() else "yolo11l.pt"
    cpus = os.cpu_count() or 4
    n_workers = args.workers or max(1, min(cpus - 1, 24))
    n_threads = max(1, cpus // n_workers)
    rows = {int(x) for x in args.rows.split(",")} if args.rows else None
    files = sorted(Path(args.input).glob("*.parquet"))
    log(f"{len(files)} parquet files, {n_workers} decoders x {n_threads} threads, "
        f"{args.sample_fps} frames/s")

    start = time.time()
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=n_workers * 2)
    out_q = ctx.Queue(maxsize=max(16, n_workers * 4))
    tmp_dir = tempfile.mkdtemp(dir="/dev/shm" if os.path.isdir("/dev/shm") else None)

    def spawn(wid):
        p = ctx.Process(target=decoder_main, daemon=True,
                        args=(wid, task_q, out_q, args.sample_fps, n_threads, tmp_dir))
        p.start()
        return p

    workers = [spawn(w) for w in range(n_workers)]   # start before CUDA is initialised
    detector = Detector(weights, args.half)

    submitted = set()
    reader = threading.Thread(target=reader_main, daemon=True,
                              args=(files, task_q, out_q, submitted, n_workers, args.max_videos, rows))
    reader.start()

    out = open(args.output, "w")
    dump = open(args.dump_frames, "w") if args.dump_frames else None
    videos = {}                  # vid -> state
    current = [None] * n_workers  # vid each decoder is working on
    last_msg = [time.time()] * n_workers
    finished = set()
    reader_done = False
    idle_since = None
    n_frames = 0
    prof = {"gpu": 0.0, "wait": 0.0}                 # main process
    dec = {"decode": 0.0, "convert": 0.0, "letterbox": 0.0, "send": 0.0, "decoded": 0, "kept": 0}

    def state(vid):
        return videos.setdefault(vid, {"per_sec": {}, "frames": [], "max_sec": -1, "t": time.time()})

    def write(lines):
        for line in lines:
            out.write(json.dumps(line) + "\n")
        out.flush()

    def finish(vid, lines):
        write(lines)
        st = videos.pop(vid, None)
        if dump is not None and st is not None:
            dump.write(json.dumps({"video": vid, "frames": st["frames"]}) + "\n")
        finished.add(vid)
        dt = time.time() - st["t"] if st else 0
        log(f"{vid}: {len(lines)} lines in {dt:.1f}s")

    def fail(vid, msg, last_sec):
        st = state(vid)
        lines = second_lines(vid, st["per_sec"], last_sec - 1) if last_sec > 0 else []
        lines.append({"video": vid, "error": msg})
        finish(vid, lines)

    try:
        while True:
            if reader_done and finished >= submitted:
                break
            c = time.perf_counter()
            try:
                msg = out_q.get(timeout=1.0)
            except queue.Empty:
                msg = None
            prof["wait"] += time.perf_counter() - c   # large = decoders are the bottleneck

            if msg is None:
                now = time.time()
                for w, p in enumerate(workers):
                    vid = current[w]
                    dead = not p.is_alive()
                    hung = vid is not None and now - last_msg[w] > STALL_S
                    if hung:
                        p.terminate(); p.join(5)
                    if vid is not None and (dead or hung):
                        fail(vid, "decoder crashed" if dead else "decoder timed out", state(vid)["max_sec"])
                        current[w] = None
                    if (dead and p.exitcode not in (0, None)) or hung:
                        workers[w] = spawn(w)
                        last_msg[w] = now
                # safety net: everything handed out but some videos never reported back
                if reader_done and all(c is None for c in current):
                    idle_since = idle_since or now
                    if now - idle_since > 60:
                        for vid in sorted(submitted - finished):
                            fail(vid, "video was lost", -1)
                else:
                    idle_since = None
                continue

            kind = msg[0]
            if kind == "stats":
                for k, v in msg[3].items():
                    dec[k] += v
                continue
            if kind in ("frames", "done", "error") and msg[2] in finished:
                continue  # late message from a decoder we already gave up on
            if kind == "lines":
                write(msg[1])
            elif kind == "reader_done":
                reader_done = True
            elif kind == "start":
                _, w, vid = msg
                current[w] = vid; last_msg[w] = time.time()
                state(vid)
            elif kind == "frames":
                _, w, vid, batch, secs, ts = msg
                last_msg[w] = time.time()
                c = time.perf_counter()
                counts = detector(batch)
                prof["gpu"] += time.perf_counter() - c
                n_frames += len(counts)
                st = state(vid)
                for s, t, c in zip(secs, ts, counts):
                    st["per_sec"].setdefault(s, []).append(c)
                    st["max_sec"] = max(st["max_sec"], s)
                    if dump is not None:
                        st["frames"].append([round(t, 4)] + c)
            elif kind == "done":
                _, w, vid, last_sec = msg
                current[w] = None; last_msg[w] = time.time()
                finish(vid, second_lines(vid, state(vid)["per_sec"], last_sec))
            elif kind == "error":
                _, w, vid, err, last_sec = msg
                current[w] = None; last_msg[w] = time.time()
                fail(vid, err, last_sec)
    finally:
        out.close()
        if dump:
            dump.close()
        for p in workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    el = time.time() - start
    log(f"done: {len(finished)} videos, {n_frames} frames through the model in {el:.1f}s "
        f"({n_frames / max(el, 1e-9):.1f} frames/s)")
    # --- where the time goes ---
    log(f"[profile] main process: model {prof['gpu']:.1f}s, waiting for frames {prof['wait']:.1f}s, "
        f"other {el - prof['gpu'] - prof['wait']:.1f}s (wall {el:.1f}s)")
    tot = sum(dec[k] for k in ("decode", "convert", "letterbox", "send"))
    log(f"[profile] decoders (summed over {n_workers}): decode {dec['decode']:.1f}s, "
        f"convert {dec['convert']:.1f}s, letterbox {dec['letterbox']:.1f}s, "
        f"blocked sending {dec['send']:.1f}s; {dec['decoded']} frames decoded "
        f"({dec['decoded'] / max(dec['decode'], 1e-9):.0f}/s per decoder), {dec['kept']} kept; "
        f"busy {tot / max(el * n_workers, 1e-9) * 100:.0f}% of decoder time")


if __name__ == "__main__":
    main()
