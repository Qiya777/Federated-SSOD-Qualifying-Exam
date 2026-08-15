# Code Overview
This repository contains the main implementation of our federated semi-supervised object detection framework.

## Main Files
### V1_Navigation.py — Frame-Level Federated
Implements the frame-level federated baseline with single-frame pseudo-label learning and federated aggregation.

### V2_Tracking.py — Temporal Refinement
Extends V1 by using temporal information from neighboring frames to refine client-side pseudo labels.

### V4_B_r001_to_r100.py — Full Method
Extends V2 with the adaptive server correction mechanism for server-guided global aggregation.

## Method Progression
V1_Navigation  
→ Frame-Level Federated
V2_Tracking  
→ + Temporal Refinement
V4_B_r001_to_r100  
→ + Server Correction (Full Method)

## EfficientTeacher/
The `EfficientTeacher/` directory provides the underlying semi-supervised object detection framework used by our federated experiments.
It includes the model architecture, SSOD training pipeline, pseudo-label generation, loss functions, dataset/configuration files, and other supporting utilities.
Our federated, temporal refinement, and server-side correction methods are implemented on top of this framework.