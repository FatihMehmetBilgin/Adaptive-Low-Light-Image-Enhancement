# Adaptive Low-Light Image Enhancement

This repository contains the official Python implementation of the CPU-only, parallel hybrid CLAHE-Retinex pipeline for low-light image enhancement, operating in the HSV color space.

## Features
* **Parallel Architecture:** Runs CLAHE and Single-Scale Retinex (SSR) independently on the Value (V) channel.
* **Color Fidelity:** Hue (H) and Saturation (S) channels are bypassed to prevent color shifts.
* **Entropy-Based Fusion:** Automatically calculates fusion weights based on Shannon entropy, requiring no manual tuning.
* **CPU-Optimized:** Designed for thermally constrained edge devices without needing a GPU.

## Requirements
* Python 3.14+
* NumPy
* SciPy
* OpenCV (`opencv-python`)
* scikit-image
* brisque

## Dataset Setup
We evaluate on the **LOL (Low-Light) dataset**. Due to size limits, the dataset is not included in this repository. 
1. Download the LOL dataset.
2. Create a `data/lol` folder in the root directory.
3. Place the `our485` and `eval15` splits inside the `data/lol` folder.

## Usage
To run the enhancement pipeline on the eval15 test split and calculate PSNR, SSIM, and BRISQUE metrics:
```bash
python main.py
