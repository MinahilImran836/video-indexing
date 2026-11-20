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
