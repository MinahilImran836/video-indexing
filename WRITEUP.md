# Write-up

## 1. Your timings (100 words)
The baseline was tested for 20 videos only not all 50 files and the estimated time for all 50 files from them was ~88 hours. The final version using 2 T4 GPUs processed all files in 11,770 seconds (3.27 hours). Reduciug the frame sampling from all to 5 frames per second saved most of the time.

## 2. Your pipeline (150 words)
Each Parquet file is opened and the mp4 column which contains the video binary is read by several CPU decoder processes so that decoding is done in parallel. Each decoder reads the video frame by frame but only 5 frames are kept for the detection. they are resized and sent in batches of 32 to the main process. Then YOLO11l runs on GPU using FP16 to detect the objects and decide which object appears in at least half the sample frames and calculates the median people count and writes the output in JSONL.

## 3. Where the time goes (150 words)
GPU inference consumes most of the time. During profiling it was noted that GPU inference consumed ~80% of runtime. The CPU decoder processes decode videos, select frames and then resize them. They spent a lot of time blocked because the GPU queue was full. 

## 4. Checking correctness (100 words)
The first 20 baseline results were kept to compare against the faster version the baseline had results 0.9870, 96.23% and 99.69 for object detection F1, people count accuracy and people count accuracy within 1 respectively. and the fast version was also tested on additional 10 videos from the parquet files 25 and 49 they also achieved almost same results as first 20 videos.

## 5. Working with AI (100 words)
The AI initially suggested that CPU decoding might be the one slowing down the process, so to measure it profiling was added to measure GPU inference and CPU decoding, frame resizing and queue waiting which revealed that GPU inference was the one actually consuming most of the time.

## 6. Next steps (100 words)
Since in the evaluated pipeline GPU inference was taking up most of the time, I would find more ways to process batches more efficiently to reduce this time and since only 5 frames per second were chosen for this pipeline I would test more sampling rates and more variety of videos to see if we could improve accuracy and improve time 
