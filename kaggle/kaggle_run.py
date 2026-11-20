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
    args = ap.parse_args()

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

    print(f"{len(files)} files, {len(done)} already done", flush=True)
    for f in files:
        stem = Path(f).stem
        if stem in done:
            continue
        in_dir, out_dir = WORK / "in", WORK / "out"
        shutil.rmtree(in_dir, ignore_errors=True)
        in_dir.mkdir(parents=True); out_dir.mkdir(parents=True, exist_ok=True)

        t_dl = time.time()
        local = hf_hub_download(DATASET, f, repo_type="dataset", token=read_tok, local_dir=WORK / "dl")
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
        # upload frames first, part last: the part file is the "done" marker
        for p, dest in reversed(uploads):
            api.upload_file(path_or_fileobj=str(p), path_in_repo=dest, repo_id=args.repo,
                            repo_type="dataset", token=write_tok)
        api.upload_file(path_or_fileobj=json.dumps({"download_s": t_dl, "run_s": t_run}).encode(),
                        path_in_repo=f"{args.tag}/times/{stem}.json", repo_id=args.repo,
                        repo_type="dataset", token=write_tok)
        shutil.rmtree(in_dir, ignore_errors=True)
        for p, _ in uploads:
            p.unlink()
        print(f"[done] {stem}: download {t_dl:.0f}s, run {t_run:.0f}s", flush=True)


def merge(api, args, files, tok):
    out = REPO_DIR / "results.jsonl.gz"
    total = 0.0
    with gzip.open(out, "wt") as g:
        for f in files:
            stem = Path(f).stem
            p = hf_hub_download(args.repo, f"{args.tag}/parts/{stem}.jsonl", repo_type="dataset",
                                token=tok, local_dir=WORK / "merge")
            g.write(Path(p).read_text())
            t = hf_hub_download(args.repo, f"{args.tag}/times/{stem}.json", repo_type="dataset",
                                token=tok, local_dir=WORK / "merge")
            total += json.loads(Path(t).read_text())["run_s"]
    print(f"wrote {out}; sum of run.py times = {total / 3600:.2f} h")


if __name__ == "__main__":
    main()
