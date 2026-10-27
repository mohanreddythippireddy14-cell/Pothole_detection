# 09. Reproducing Results

## Environment
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Set `POTHOLE_DATA_ROOT` to store datasets outside the repo. Kaggle downloads need Kaggle API credentials.

## Running the pipeline
Put `yolov8n_pothole_v2_final_best.pt` and `yolov8n_seg_pothrgbd_best.pt` in `checkpoints/` (from the Releases page, or train them), then:
```bash
python scripts/10_full_pipeline.py --images docs/assets/samples/potholes103_raw.png --out-dir outputs/
```
Add `--no-depth` to skip Depth-Anything-V2 (faster; severity from size only). Both modes run on CPU; a GPU is faster.

## Training order
`01 → 02 → 03 → 04` (v1 detector) · `06 → 07 → 08` (segmenter) · `09` (depth check) · `11 → 12 → 13` (v2 detector) · `14` (severity calibration). Full table in the main README.

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```
