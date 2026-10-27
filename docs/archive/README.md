# Archive — historical working notes

These files are working notes and session logs from earlier development. They are kept
for reference and future work, but they are **not maintained** and may be out of date.

Known corrections (see `FUTURE_ACTIONS.md` at the repo root and the main `README.md`):

- The detection mAP@50 of **0.95** quoted here is **not valid**: it was measured on the
  original 67-image v1 test set, whose higher-resolution copies were part of the v2
  training data. The 5-fold cross-validation result (mAP@50 0.73 ± 0.01) is the
  reliable figure until a clean re-evaluation is done.
- References to RDD2022, v2-2022, 2,083 images and a 70/15/15 split with a 312-image
  test set do not match the code. The pipeline actually merges andrewmvd (replaced by
  chitholian's 720×720 copies), rajdalsaniya and banilkumar (background images) into
  1,986 images, evaluated with 5-fold CV plus a 90/10 final split.
- Statements that there is "no data leakage" are not correct (see above).

| File | What it is |
|---|---|
| `CHANGELOG.md` | Session-by-session development log |
| `PROJECT_AUDIT.md` | An earlier self-audit of the project |
| `_state/` | Documentation-generation working state (interview notes, inventory, fact ledger, progress) |
