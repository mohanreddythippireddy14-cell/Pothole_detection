# 04. Phase 2: Instance Segmentation

## Architectural Shift
To solve the geometric distortion identified in Phase 1, the pipeline was upgraded to utilize `YOLOv8n-Seg`. Instead of relying on a crude rectangular boundary, the model now predicts a pixel-perfect polygon mask for each detected instance.

## Implementation Details
Due to the computational overhead of high-resolution instance segmentation on wide field-of-view images, the pipeline employs a two-stage approach:
1. **Detection:** A rapid YOLOv8n pass identifies the general bounding region.
2. **Crop & Segment:** The bounding box is cropped (with padding), and only that specific tensor is passed to YOLOv8n-Seg for precise mask extraction.

## Results
The segmentation model achieved a **Mask mAP@50 of 0.908**. This pixel-level accuracy allowed for the exact calculation of the pothole's 2D surface area relative to the frame, forming the foundation of the geometric feature vector.

```mermaid
flowchart LR
    %% Styling
    classDef phase1 fill:#ffebee,stroke:#c62828,stroke-width:2px;
    classDef phase2 fill:#e8eaf6,stroke:#283593,stroke-width:2px;
    classDef component fill:#fff,stroke:#333,stroke-width:1px;

    subgraph P1 [Phase 1: Bounding Box Baseline]
        direction LR
        A1([🖼️ RGB Image]):::component --> B1[🧠 YOLOv8n Object Det.]:::phase1
        B1 --> C1[🟦 Bounding Box]:::component
        C1 --> D1[⚠️ Box-Based Severity]:::phase1
    end

    subgraph P2 [Phase 2: Segmentation & Depth Evolution]
        direction LR
        A2([🖼️ RGB Image]):::component --> B2[🧠 YOLOv8n Object Det.]:::phase2
        B2 --> C2[🟦 Bounding Box]:::component
        C2 --> D2[✂️ Crop + Pad]:::component
        D2 --> E2[🎨 YOLOv8n-Seg]:::phase2
        E2 --> F2[💠 Pixel Mask]:::component
        
        A2 -.-> G2[📏 Depth-Anything-V2]:::phase2
        G2 -.-> H2[🗺️ Depth Map]:::component
        
        F2 --> I2[🎯 Masking / Overlay]:::component
        H2 -.-> I2
        I2 --> J2[📊 Mask + Depth Feature Vector]:::component
        J2 --> K2[⚙️ ASTM D6433 Severity Engine]:::phase2
    end

```
