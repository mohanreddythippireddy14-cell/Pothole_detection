# 01. System Overview

## The Municipal Context
Road departments are tasked with identifying and repairing damage across vast geographic areas. Manual inspection is expensive, slow, and prone to subjective severity assessments. Citizen reporting (e.g., 311 calls) is valuable but results in duplicate reports and lacks quantitative severity metrics for triage.

## The Objective
This ML component serves as the automated ingestion and triage engine for a larger **SIH-style Municipal Dashboard**. It processes raw RGB images (e.g., from dashcams, traffic cameras, or citizen apps) and extracts actionable, structured data to prioritize road repair teams.

## Core Contribution
The core technical contribution of this project is not the invention of a novel model architecture, but rather the **unique integration of disparate subsystems** to solve a real-world constraint: the lack of accessible 3D road damage data. 
By fusing state-of-the-art 2D instance segmentation (YOLOv8n-Seg) with continuous monocular depth estimation (Depth-Anything-V2), we project 2D pixels into a pseudo-3D heuristic space to satisfy ASTM D6433 severity guidelines.
