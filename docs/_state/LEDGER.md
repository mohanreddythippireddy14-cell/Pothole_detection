| ID | Fact | Value | Source File(s) | Source Hash/mtime | Status | Date |
|---|---|---|---|---|---|---|
| F01 | Architecture Pipeline | YOLOv8n (detect) -> YOLOv8n-seg (mask) -> DA-V2 (depth) -> Severity heuristic | `context/PROJECT.md`, `scripts/10_full_pipeline.py` | (current) | VERIFIED | 2026-09-30 |
| F02 | Phase 1 v1 Metric | mAP@50 = 0.77, mAP@50:95 = 0.48, Recall = 0.68 | `context/PROJECT.md` | (current) | VERIFIED | 2026-09-30 |
| F03 | Phase 1 v2 Metric | mAP@50 = 0.95, mAP@50:95 = 0.70, Recall = 0.86 | `context/PROJECT.md` | (current) | VERIFIED | 2026-09-30 |
| F04 | Phase 2 Metric | Mask mAP@50 = 0.908, mAP@50:95 = 0.592 | `context/PROJECT.md` | (current) | VERIFIED | 2026-09-30 |
| F05 | Dataset 1 License | andrewmvd/pothole-detection is DbCL v1.0 | Web Search | (N/A) | VERIFIED | 2026-09-30 |
| F06 | Dataset 2 License | PothRGBD is MIT | Web Search | (N/A) | VERIFIED | 2026-09-30 |
| F07 | Dataset 3 License | idanbaru/annotated-potholes-with-severity-levels is ODbL | Web Search | (N/A) | VERIFIED | 2026-09-30 |
| F08 | Data Leakage | None (test set completely held out) | User interview | (N/A) | USER-STATED | 2026-09-30 |
| F09 | Prior Work/References | Custom work, no papers referenced specifically | User interview | (N/A) | USER-STATED | 2026-09-30 |
| F10 | Inference Type | Offline image predictor only (real-time is future work) | User interview | (N/A) | USER-STATED | 2026-09-30 |
| F11 | Datasets & Cache Path | Kept on external drive (POTHOLE_DATA_ROOT) to save space | `src/config.py`, User instruction | (current) | VERIFIED | 2026-09-30 |
| F12 | Severity Heuristics | Based on ASTM D6433 logic (depth escalates geometric severity) | `scripts/10_full_pipeline.py` | (current) | VERIFIED | 2026-09-30 |
| F13 | Project Motivation | ML engine for real-time municipal road damage dashboard | `docs/_state/INTERVIEW.md` | (current) | USER-STATED | 2026-09-30 |
| F14 | Phase 1 Failure Mode | Low recall (0.66) due to small dataset size and bounding box distortion | `docs/_state/INTERVIEW.md` | (current) | VERIFIED | 2026-09-30 |
| F15 | Target Audience | Computer Vision Research Labs | `docs/_state/INTERVIEW.md` | (current) | USER-STATED | 2026-09-30 |
| F16 | Phase 2 Split | 70/15/15 randomized split on 2,083 images | `docs/_state/INTERVIEW.md` | (current) | USER-STATED | 2026-09-30 |
| F17 | Core Contribution | Unique integration of disparate subsystems + applied severity scaling | `docs/_state/INTERVIEW.md` | (current) | USER-STATED | 2026-09-30 |
| S01 | Sample Image 1 | `potholes103_annotated.png` (predicted pipeline output) | `docs/assets/samples/` | (current) | UNVERIFIED (needs visual check) | 2026-09-30 |
| S02 | Sample Image 2 | `potholes120_annotated.png` (predicted pipeline output) | `docs/assets/samples/` | (current) | UNVERIFIED (needs visual check) | 2026-09-30 |
| S03 | Sample Image 3 | `potholes122_annotated.png` (predicted pipeline output) | `docs/assets/samples/` | (current) | UNVERIFIED (needs visual check) | 2026-09-30 |
| S04 | Sample Image 4 | `potholes125_annotated.png` (predicted pipeline output) | `docs/assets/samples/` | (current) | UNVERIFIED (needs visual check) | 2026-09-30 |
| S05 | Sample Image 5 | `potholes127_annotated.png` (predicted pipeline output) | `docs/assets/samples/` | (current) | UNVERIFIED (needs visual check) | 2026-09-30 |
| S06 | Sample Image 6 | `potholes138_annotated.png` (predicted pipeline output) | `docs/assets/samples/` | (current) | UNVERIFIED (needs visual check) | 2026-09-30 |
