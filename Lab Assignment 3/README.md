# Lab Assignment 3 - YOLOv8 Object Detection, Segmentation, and Classification

## Overview

This lab implements a complete multi-task computer vision pipeline using YOLOv8. It demonstrates:

- Object detection (YOLOv8n)
- Instance segmentation (YOLOv8n-seg)
- Image classification (YOLOv8n-cls)
- Benchmarking across model variants
- Hyperparameter experiments and fine-tuning
- Export to deployment formats (ONNX and TorchScript)

The notebook is designed for Google Colab with GPU support and includes evaluation metrics, plots, and a real-world case study.

## Notebook

- YOLOv8_Object_Detection_Lab.ipynb

## Learning Objectives

- Understand YOLOv8 architecture and task variants
- Build an end-to-end detection and segmentation workflow
- Evaluate models using standard metrics (mAP, precision, recall)
- Compare speed and parameter count across YOLOv8 variants
- Export trained models for deployment

## Tasks Implemented (Detailed)

1. Environment setup and GPU check
2. Dependency installation and imports
3. Dataset selection and justification (COCO128 default)
4. Dataset download, EDA, and sample visualization
5. YOLOv8 detection model training
6. Detection evaluation and training curves
7. Inference on test image and detection visualization
8. YOLOv8 segmentation inference and mask overlay
9. YOLOv8 classification inference and Top-5 results
10. Variant benchmarking (n, s, m)
11. Hyperparameter experiments and fine-tuning with augmentation
12. Export to ONNX and TorchScript
13. Real-world application case study
14. Results summary and observations

## Dataset

- Default: COCO128 (Ultralytics built-in dataset)
- Size: 128 images, 80 classes
- Reason: fast download, diverse objects, good for lab-scale runs

If you use your own dataset, update the `DATA_YAML` path in the notebook.

## Models Used

- Detection: yolov8n
- Segmentation: yolov8n-seg
- Classification: yolov8n-cls

## Key Hyperparameters (Baseline)

- epochs: 10
- imgsz: 640
- batch: 16
- lr0: 0.01

These values are selected for COCO128 and can be increased for larger datasets.

## Outputs Generated

- Training curves (loss, mAP, precision/recall)
- Detection results image
- Segmentation output and mask overlay
- Top-5 classification chart
- Variant benchmark plots
- Exported model files and size comparison

Typical output paths (Colab):

- /content/runs/...
- /content/predictions/...

## How To Run (Colab)

1. Open the notebook in Colab.
2. Set runtime to GPU (T4 recommended).
3. Run cells in order from top to bottom.
4. Review plots, metrics, and saved outputs.

## How To Run (Local)

1. Create a virtual environment.
2. Install dependencies listed below.
3. Ensure you have a compatible GPU and CUDA drivers (optional but recommended).
4. Run the notebook in Jupyter or VS Code.

## Requirements

- Python 3.9+
- ultralytics==8.2.0
- torch
- opencv-python-headless
- supervision
- numpy
- pandas
- matplotlib
- seaborn
- roboflow (optional for custom dataset download)

## Notes

- The notebook includes a PyTorch 2.6 compatibility patch for `torch.load`.
- COCO128 will download automatically on first run.
- For custom datasets, update YAML paths and class names.

## Troubleshooting

- No GPU detected: enable GPU runtime in Colab.
- Dataset not found: re-run dataset download cell or update `DATA_YAML`.
- Missing dependencies: re-run the install cell.
- Slow training: reduce image size or batch size, or use yolov8n.

## Results (Example)

- Detection mAP@50: ~0.60 on COCO128 (10 epochs)
- Inference time: ~8 ms (YOLOv8n on T4 GPU)

These values are indicative and will vary with environment and training settings.

## Repository Structure

- YOLOv8_Object_Detection_Lab.ipynb
- README.md

## Author

Rohit Vishwas Thorat
