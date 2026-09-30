# 08. Deployment & Dashboard Integration

## Context
This repository represents the ML ingestion engine. In production, this pipeline operates as a microservice asynchronously processing video frames or image uploads.

## Data Flow
Upon processing a frame, the engine outputs an annotated image (for visual confirmation) and a structured JSON payload containing:
- Pothole count
- Bounding box coordinates
- Severity assignments (`LOW`, `MEDIUM`, `HIGH`)
- Confidence scores

This payload is pushed to the SIH Municipal Dashboard's PostgreSQL/PostGIS database, where it is aggregated geographically. If 10 citizens upload images of the same pothole, the dashboard clusters them based on GPS metadata, utilizing this ML engine to confirm the damage and assess its priority.
