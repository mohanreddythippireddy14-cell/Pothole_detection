# 02. Dataset Curation and Strict Splitting

## The Low Recall Problem
Initial baseline testing (Phase 1 v1) using a standard 665-image dataset yielded high precision but abysmal recall (0.66). The model was overly conservative, missing 34% of real-world potholes due to insufficient lighting and environmental diversity in the training distribution.

## Dataset Merging
To resolve this, we aggregated four disparate public datasets:
1. **andrewmvd/pothole-detection** (Pascal VOC format)
2. **v2-2022 Pothole Dataset** (YOLO format)
3. **RDD2022** (Filtering strictly for class `D40` - potholes)
4. **PothRGBD** (Providing mask polygons)

All datasets were normalized to standard YOLO object/segmentation format, mapping the target class universally to ID `0`. Deduplication via hash-based similarity yielded a final curated corpus of **2,083 images**.

## Strict Evaluation Protocol
To prevent data leakage and ensure unbiased quantitative evaluation:
- **70% Training (~1,458 images):** Used exclusively for model gradient updates and augmentations.
- **15% Validation (~312 images):** Used strictly for hyperparameter tuning and early stopping.
- **15% Held-out Test (~313 images):** Locked away until final evaluation.

```mermaid
flowchart TD
    %% Styling
    classDef ds fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef script fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef split fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;
    classDef result fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;

    subgraph Data Sources
        direction LR
        DS1[(andrewmvd)]:::ds
        DS2[(v2-2022)]:::ds
        DS3[(RDD2022)]:::ds
        DS4[(PothRGBD)]:::ds
    end

    subgraph Data Cleaning & Parsing
        direction LR
        DS1 --> CP[⚙️ Parse XML to YOLO txt]:::script
        DS2 --> CP2[⚙️ Map 'pothole' string to ID 0]:::script
        DS3 --> CP3[⚙️ Filter subset: D40 labels only]:::script
        DS4 --> CP4[⚙️ Format mask polygons]:::script
    end

    CP --> MERGE[🔄 Deduplication & Merge]:::script
    CP2 --> MERGE
    CP3 --> MERGE
    
    MERGE --> SPLIT{Strict 70/15/15 Split}:::split
    
    SPLIT -->|70%| TRAIN[(Training Set)]:::result
    SPLIT -->|15%| VAL[(Validation Set)]:::result
    SPLIT -->|15%| TEST[(Held-out Test Set)]:::result
    
    TRAIN -.->|⛔ No Leakage| TEST
    VAL -.->|⛔ No Leakage| TEST
    
    TRAIN --> MODEL[🧠 YOLO Fine-Tuning]:::script
    VAL --> MODEL
    TEST --> EVAL[📊 Unbiased Quantitative Evaluation]:::result

```
