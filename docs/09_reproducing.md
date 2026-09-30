# 09. Reproducing Results

## Environment Setup
1. Clone the repository.
2. Initialize the virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Running the Pipeline
To execute the end-to-end Phase 3 pipeline on sample images:
```bash
python scripts/10_full_pipeline.py \
    --images docs/assets/samples/potholes103_raw.png \
    --out-dir outputs/ \
    --no-depth
```
*(Note: Remove `--no-depth` to execute the heavy Depth-Anything-V2 branch. Requires CUDA).*

## Sample Outputs
See `docs/assets/diagrams/05_inference_example.md` for a visual comparison of inputs to pipeline outputs.
