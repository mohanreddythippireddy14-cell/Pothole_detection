# AI-Powered Road Damage Detection & Municipal Management System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview
This repository contains the Machine Learning vision pipeline for a real-time municipal dashboard system designed to autonomously detect, localize, and assess the severity of road damage (potholes).

While many object detection projects stop at drawing bounding boxes, true municipal triage requires geometric understanding. This pipeline uniquely integrates:
1. **Instance Segmentation (YOLOv8n-Seg):** To extract pixel-perfect damage area geometries.
2. **Monocular Depth Estimation (Depth-Anything-V2):** To approximate pothole depth without stereo cameras or LiDAR.
3. **ASTM D6433 Heuristics Engine:** To cross-reference segmented area and depth maps against established civil engineering standards, assigning functional `LOW`, `MEDIUM`, or `HIGH` severity ratings.

## Quick Links
- **[Documentation Suite](docs/01_overview.md)**: Comprehensive architectural breakdown.
- **[Model Card](docs/MODEL_CARD.md)**: Evaluation metrics, dataset details, and known failure modes.
- **[Open Items](OPEN_ITEMS.md)**: Current limitations and research directions.

## Pipeline Architecture
```mermaid
flowchart TD
    %% Styling
    classDef input fill:#f9f9f9,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;
    classDef model fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef decision fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;

    %% Nodes
    A([📸 Raw Input Image]):::input --> B[🧠 YOLOv8n Object Detection]:::model
    
    subgraph Detection Phase
        B --> C{Potholes Detected?}:::decision
        C -- No --> D([✅ Output Safe Road]):::output
        C -- Yes --> E[✂️ Crop & Pad Bounding Boxes]:::process
    end
    
    subgraph Segmentation & Depth Phase
        E --> F[🎨 YOLOv8n-Seg Instance Segmentation]:::model
        F --> G[💠 Extract Binary Pothole Mask]:::process
        
        A -.-> H[📏 Depth-Anything-V2 Monocular Depth]:::model
        H --> I[🗺️ Continuous Relative Depth Map]:::process
        
        G --> J[🎯 Overlay Mask on Depth Map]:::process
        I --> J
    end
    
    subgraph Severity Engine
        J --> K[🧮 Calculate Depth & Geometric Features]:::process
        K --> L[⚙️ ASTM D6433 Heuristic Engine]:::model
        L --> M{Assign Severity}:::decision
        M -->|Low / Medium / High| N[🏷️ Annotate Output Frame]:::process
    end
    
    N --> O[(📊 Push to Admin Dashboard)]:::output

```

## Setup & Inference
See [docs/09_reproducing.md](docs/09_reproducing.md) for full environment setup and execution instructions.
