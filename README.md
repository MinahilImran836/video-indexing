# Video Indexing at Speed

For each second of each FineVideo video, records which of five objects
(person, car, dog, laptop, cell phone) YOLO11l finds and how many people are visible.

## Install

No `apt-get` packages are needed (the OpenCV wheel includes FFmpeg).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparation command (run once, needs internet, not timed)

```bash
cd <repository folder>
MPLBACKEND=Agg python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

This downloads `yolo11l.pt` into the repository folder, where `run.py` looks for it.
(`MPLBACKEND=Agg` only stops matplotlib from looking for a display; nothing is plotted.)

## Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

## Repository layout

| Path | What it is |
|---|---|
| `run.py` | The script |
| `requirements.txt` | Exact package versions |
| `results.jsonl.gz` | Results for the first 50 FineVideo files |
| `HARDWARE.md` | Where and how long the full run took |
| `WRITEUP.md` | Answers to the write-up questions |
| `ai_chats/` | Full AI chat history |
| `kaggle/` | Driver used to run the 50 files on Kaggle with checkpoints |
| `tools/` | Correctness checks used during development |
