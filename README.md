# ICEL

This repository contains an implementation of the ICEL (Inconsistent Explanations) method from the paper "ICEL: Learning with Inconsistent Explanations". The implementation is tested on the Tiny ImageNet dataset using a ResNet-18 architecture. The project includes scripts for training, validation, and testing on a single GPU.

## Table of Contents
- [Requirements](##requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Validation](#validation)
  - [Testing](#testing)
- [References](#references)

## Requirements

- Python 3.7+
- PyTorch 1.7.0+
- torchvision 0.8.0+
- CUDA (for GPU support)
- Other Python dependencies (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/elahehale/ICEL.git
    cd ICEL
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the Tiny ImageNet dataset and place it in the appropriate directory structure:
    ```
    tiny-imagenet-200/
    └── data/
        ├── train/
        ├── val/
        └── test/
    ```

## Project Structure

- **`dataset.py`**: Contains the custom dataset class `TinyImageNetValDataset` for loading the Tiny ImageNet validation set.
- **`model.py`**: Defines the ResNet-18 model and functions for loading checkpoints.
- **`train.py`**: Contains the training, validation, and testing functions.
- **`utils.py`**: Includes utility functions such as argument parsing, seed setting, and image saving.
- **`main.py`**: The main script to run training, validation, or testing.
- **`requirements.txt`**: lists all the dependencies needed for the project.
- **`checkpoints/`**: Directory to save model checkpoints.
- **`logs/`**: Directory to save TensorBoard logs.

