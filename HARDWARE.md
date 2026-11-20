# Hardware

The full run on all 50 files was done on a **Kaggle notebook** (free tier), using the "GPU T4 x2" accelerator.

- **Where:** Kaggle notebook (free GPU tier), run as a background "Save & Run All" job
- **GPU (model, memory):** 2× NVIDIA Tesla T4, 15360 MiB (15 GB) each. The 50 files were split between the two GPUs, 25 per GPU, with one `run.py` process per GPU.
- **CPU (model, cores):** Intel(R) Xeon(R) CPU @ 2.00GHz, 4 vCPUs
- **RAM:** 31 GB
- **Disk (SSD / hard disk):** Kaggle cloud VM disk (overlay filesystem). Each Parquet file was downloaded, processed and deleted, so at most one file per GPU was on disk at a time.
- **Operating system:** Linux (Kaggle Docker image), Python 3.13
- **Full run time (start to finish, all 50 files):** **11,770 s = 3 h 16 min** on both GPUs together, including downloading each Parquet file from Hugging Face. The sum of the `run.py` times was 23,171 s (6.44 h), which is what one T4 alone would need.

Output: 50 files, 1,650 videos, 461,757 second-lines (128.3 h of video), 0 error lines.

**Software note:** the full run used Kaggle's preinstalled packages (Python 3.13, torch 2.11.0+cu130,
opencv 4.14.0, numpy 2.1.3) with ultralytics 8.3.40. `requirements.txt` pins the versions for Python 3.11
(torch 2.5.1 CUDA 12 build, opencv 4.10.0.84, numpy 1.26.4). I tested those pinned versions in a fresh Python 3.11
environment on the same 10 reference videos. Every sampled frame had the same timestamp, 97.4% of frames had identical
counts, and the per-second scores against the every-frame reference were the same
(F1 0.9865 vs 0.9870, exact 96.1% vs 96.2%, within-1 99.7% vs 99.7%).
