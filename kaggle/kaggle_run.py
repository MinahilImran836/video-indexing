"""Kaggle driver: process FineVideo one Parquet file at a time, with checkpoints.

For each of the 50 files:
  download it -> run run.py on it -> upload the result part to your private
  Hugging Face dataset repo -> delete the local file.
Files whose part is already in the repo are skipped, so after a disconnect
you just run this again and it continues where it stopped.

Usage inside a Kaggle notebook cell:
  !python kaggle/kaggle_run.py --repo YOURNAME/finevideo-results --tag baseline \
        --files 2 --run-args "--max-videos 10 --dump-frames"
  !python kaggle/kaggle_run.py --repo YOURNAME/finevideo-results --tag final
  !python kaggle/kaggle_run.py --repo YOURNAME/finevideo-results --tag final --merge

Kaggle secrets needed (Add-ons -> Secrets): HF_TOKEN_READ, HF_TOKEN_WRITE
"""
import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url

DATASET = "HuggingFaceFV/finevideo"
FILE_RE = re.compile(r"^data/train-000[0-4][0-9]-of-01357\.parquet$")  # first 50 files
REPO_DIR = Path(__file__).resolve().parent.parent
WORK = Path("/tmp/fv")


def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ.get(name)


def check_tokens(api, read_tok, write_tok):
    """Say clearly which token works, without printing the tokens."""
    for label, tok in [("HF_TOKEN_READ", read_tok), ("HF_TOKEN_WRITE", write_tok)]:
        if not tok:
            print(f"{label}: NOT FOUND (add it in Add-ons -> Secrets and tick 'attach' for this notebook)")
            continue
        try:
            user = api.whoami(token=tok)["name"]
        except Exception as e:
            print(f"{label}: INVALID token ({type(e).__name__})")
            continue
        try:  # can this token see the gated dataset files?
            url = hf_hub_url(DATASET, "data/train-00000-of-01357.parquet", repo_type="dataset")
            get_hf_file_metadata(url, token=tok)  # HEAD request, no download
            print(f"{label}: OK (user {user}, FineVideo access OK)")
        except Exception as e:
            print(f"{label}: valid (user {user}) but NO FineVideo access ({type(e).__name__})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="your private HF dataset repo, e.g. name/finevideo-results")
    ap.add_argument("--tag", required=True, help="folder in the repo for this version, e.g. baseline / final")
    ap.add_argument("--files", type=int, default=50, help="how many of the 50 files to do")
    ap.add_argument("--file-indices", default=None, help="only these of the 50 files, e.g. 25,49")
    ap.add_argument("--script", default="run.py", help="script to run, e.g. tools/baseline_run.py")
    ap.add_argument("--run-args", default="", help="extra args for run.py")
    ap.add_argument("--merge", action="store_true", help="download all parts and build results.jsonl.gz")
    ap.add_argument("--gpus", type=int, default=1, help="split the files over this many GPUs")
    ap.add_argument("--shard", default=None, help=argparse.SUPPRESS)  # "i/n", set internally
    args = ap.parse_args()

    if args.gpus > 1 and args.shard is None and not args.merge:
        return run_parallel(args)

    # protect the reference results: "baseline*" folders may only be written by the baseline script
    if args.tag.startswith("baseline") != (args.script == "tools/baseline_run.py"):
        sys.exit(f"refusing: tag '{args.tag}' with script '{args.script}'. Use a tag starting with "
                 "'baseline' only together with --script tools/baseline_run.py (and vice versa).")

    read_tok = secret("HF_TOKEN_READ")
    write_tok = secret("HF_TOKEN_WRITE")
    api = HfApi()
    check_tokens(api, read_tok, write_tok)
    read_tok = read_tok or write_tok
    write_tok = write_tok or read_tok
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True, token=write_tok)

    remote = api.list_repo_files(args.repo, repo_type="dataset", token=write_tok)
    done = {Path(f).stem for f in remote if f.startswith(f"{args.tag}/parts/") and f.endswith(".jsonl")}

    files = sorted(f for f in api.list_repo_files(DATASET, repo_type="dataset", token=read_tok)
                   if FILE_RE.match(f))
    if args.file_indices:
        files = [files[int(i)] for i in args.file_indices.split(",")]
    else:
        files = files[: args.files]

    if args.merge:
        return merge(api, args, files, write_tok)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        files = files[i::n]
        global WORK
        WORK = WORK / f"shard{i}"

    session_start = float(os.environ.get("FV_SESSION_START", time.time()))
    todo = [f for f in files if Path(f).stem not in done]
    print(f"{len(files)} files, {len(files) - len(todo)} already done, {len(todo)} to do", flush=True)
    failed = []
    for f in todo:
        try:
            process_file(f, args, api, read_tok, write_tok, session_start)
        except Exception as e:  # keep going with the other files; a re-run retries this one
            print(f"[FAILED] {Path(f).stem}: {type(e).__name__}: {e}", flush=True)
            failed.append(Path(f).stem)
    if failed:
        sys.exit(f"{len(failed)} file(s) failed: {failed} - run the same command again to retry them")


def process_file(f, args, api, read_tok, write_tok, session_start):
    stem = Path(f).stem
    in_dir, out_dir = WORK / "in", WORK / "out"
    shutil.rmtree(in_dir, ignore_errors=True)
    in_dir.mkdir(parents=True); out_dir.mkdir(parents=True, exist_ok=True)

    t_dl = time.time()
    local = retry(lambda: hf_hub_download(DATASET, f, repo_type="dataset", token=read_tok,
                                          local_dir=WORK / "dl"), f"download of {stem}")
    shutil.move(local, in_dir / Path(f).name)
    t_dl = time.time() - t_dl

    tmp_out = out_dir / f"{stem}.jsonl.tmp"
    cmd = [sys.executable, str(REPO_DIR / args.script), "--input", str(in_dir), "--output", str(tmp_out)]
    extra = args.run_args.split()
    if "--dump-frames" in extra:  # give it a path
        extra.remove("--dump-frames")
        extra += ["--dump-frames", str(out_dir / f"{stem}.frames.jsonl")]
    t_run = time.time()
    subprocess.run(cmd + extra, check=True, cwd=REPO_DIR)
    t_run = time.time() - t_run

    final = out_dir / f"{stem}.jsonl"
    tmp_out.rename(final)  # only a complete file counts as done
    uploads = [(final, f"{args.tag}/parts/{stem}.jsonl")]
    frames = out_dir / f"{stem}.frames.jsonl"
    if frames.exists():
        uploads.append((frames, f"{args.tag}/frames/{stem}.frames.jsonl"))
    # times and frames first, part LAST: the part file is the "done" marker
    upload(api, json.dumps({"download_s": t_dl, "run_s": t_run, "session_start": session_start,
                            "finished_at": time.time()}).encode(),
           f"{args.tag}/times/{stem}.json", args.repo, write_tok)
    for p, dest in reversed(uploads):
        upload(api, str(p), dest, args.repo, write_tok)
    shutil.rmtree(in_dir, ignore_errors=True)
    for p, _ in uploads:
        p.unlink()
    print(f"[done] {stem}: download {t_dl:.0f}s, run {t_run:.0f}s", flush=True)


def retry(fn, what, tries=6):
    """Network calls can fail for a moment; try again with growing waits (5s ... 160s)."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if k == tries - 1:
                raise
            wait = 5 * 2 ** k
            print(f"{what} failed ({type(e).__name__}), retrying in {wait}s", flush=True)
            time.sleep(wait)


def upload(api, src, dest, repo, tok):
    """Upload with retries (two GPU shards may commit to the repo at the same moment)."""
    return retry(lambda: api.upload_file(path_or_fileobj=src, path_in_repo=dest, repo_id=repo,
                                         repo_type="dataset", token=tok), f"upload of {dest}")


def run_parallel(args):
    """Start one copy of this script per GPU, each on every n-th file, and wait for all."""
    start = time.time()
    procs = []
    for i in range(args.gpus):
        cmd = [sys.executable, __file__] + [a for a in sys.argv[1:]] + ["--shard", f"{i}/{args.gpus}"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), FV_SESSION_START=str(start))
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    wall = time.time() - start
    print(f"[parallel] {args.gpus} GPUs finished in {wall:.0f}s ({wall / 3600:.2f} h), exit codes {codes}",
          flush=True)
    if any(codes):
        sys.exit("a shard failed - run the same command again to continue")


def merge(api, args, files, tok):
    """Build results.jsonl.gz only if ALL parts are present and valid."""
    remote = set(api.list_repo_files(args.repo, repo_type="dataset", token=tok))
    stems = [Path(f).stem for f in files]
    missing = [s for s in stems if f"{args.tag}/parts/{s}.jsonl" not in remote]
    if missing:
        sys.exit(f"NOT merging: {len(missing)} of {len(stems)} files not done yet: {missing}\n"
                 f"Run the processing command again; it continues with the missing files.")

    tmp = REPO_DIR / "results.jsonl.gz.tmp"
    n_videos = n_secs = n_err = 0
    problems = []
    sessions = {}   # session_start -> last finished_at
    run_total = 0.0
    with gzip.open(tmp, "wt") as g:
        for stem in stems:
            p = retry(lambda: hf_hub_download(args.repo, f"{args.tag}/parts/{stem}.jsonl",
                                              repo_type="dataset", token=tok, local_dir=WORK / "merge"),
                      f"download of part {stem}")
            text = Path(p).read_text()
            v, sc, er, pr = check_part(stem, text)
            n_videos += v; n_secs += sc; n_err += er; problems += pr
            g.write(text if text.endswith("\n") or not text else text + "\n")
            t_path = f"{args.tag}/times/{stem}.json"
            if t_path in remote:
                t = json.loads(Path(hf_hub_download(args.repo, t_path, repo_type="dataset", token=tok,
                                                    local_dir=WORK / "merge")).read_text())
                run_total += t["run_s"]
                if "session_start" in t:
                    k = t["session_start"]
                    sessions[k] = max(sessions.get(k, 0), t["finished_at"])
    if problems:
        tmp.unlink()
        sys.exit("NOT merging, problems found:\n  " + "\n  ".join(problems[:30]))

    out = REPO_DIR / "results.jsonl.gz"
    tmp.rename(out)   # only a complete, checked file gets the real name
    print(f"OK: wrote {out}")
    print(f"  {len(stems)} files, {n_videos} videos, {n_secs} second-lines "
          f"({n_secs / 3600:.1f} h of video), {n_err} error lines")
    print(f"  sum of run.py times: {run_total:.0f}s ({run_total / 3600:.2f} h)")
    if sessions:
        wall = sum(end - start for start, end in sessions.items())
        print(f"  wall-clock over {len(sessions)} session(s): {wall:.0f}s ({wall / 3600:.2f} h) "
              f"<- use this in HARDWARE.md")
    # keep a copy next to the parts as a backup
    upload(api, str(out), f"{args.tag}/results.jsonl.gz", args.repo, tok)
    print(f"  backup uploaded to {args.repo}/{args.tag}/results.jsonl.gz")


def check_part(stem, text):
    """Each line valid JSON with the right fields; per video, seconds are 0..N with no gaps
    or duplicates; video ids belong to this file. Returns (videos, seconds, errors, problems)."""
    problems = []
    secs, errs = {}, set()
    for n, line in enumerate(text.splitlines(), 1):
        try:
            d = json.loads(line)
        except Exception:
            problems.append(f"{stem} line {n}: not valid JSON")
            continue
        vid = d.get("video", "")
        if not vid.startswith(stem + ".parquet:"):
            problems.append(f"{stem} line {n}: wrong video id {vid!r}")
        if "error" in d:
            errs.add(vid)
            continue
        if not (isinstance(d.get("second"), int) and isinstance(d.get("objects"), list)
                and isinstance(d.get("people"), int)):
            problems.append(f"{stem} line {n}: missing/invalid fields")
            continue
        secs.setdefault(vid, []).append(d["second"])
    for vid, ss in secs.items():   # (a video that failed partway stops early, but has no gaps either)
        if sorted(ss) != list(range(len(ss))):
            problems.append(f"{vid}: seconds are not 0..N without gaps/duplicates")
    videos = set(secs) | errs
    rows = sorted(int(v.rsplit(":", 1)[1]) for v in videos if v.rsplit(":", 1)[1].isdigit())
    if rows and rows != list(range(rows[-1] + 1)):
        problems.append(f"{stem}: some row numbers are missing: {sorted(set(range(rows[-1] + 1)) - set(rows))[:10]}")
    return len(videos), sum(len(v) for v in secs.values()), len(errs), problems


if __name__ == "__main__":
    main()
