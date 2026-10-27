# 03. Phase 1: Detection

## v1 baseline
YOLOv8n (COCO-pretrained) fine-tuned on andrewmvd (`scripts/03`): 100 epochs requested, early-stopped at 86. On the held-out 67-image test split: precision 0.794, recall 0.677, mAP@50 0.771, mAP@50:95 0.477.

Two problems were found: recall was low, and Ultralytics' `close_mosaic` phase (mosaic augmentation switched off for the last epochs) never ran because early stopping fired first.

## v2
Trained on the 1,986-image merged dataset with `epochs=60, patience=25, close_mosaic=15` so the clean-image phase always runs. Result (5-fold CV): precision 0.781, recall 0.642, mAP@50 0.731, mAP@50:95 0.432. These validation folds include 650 background images and harder close-ups, so they are not directly comparable to the v1 test number.

## Why boxes alone aren't enough for severity
A bounding box includes undamaged road around irregular or diagonal potholes, so box area overstates damaged area. This motivated per-crop segmentation (Phase 2). The size of this effect has not been measured.
