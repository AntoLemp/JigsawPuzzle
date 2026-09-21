# Jigsaw Puzzle Reconstruction

## Overview
This project automatically reconstructs scrambled jigsaw puzzles from square image tiles. The pieces are randomly shuffled and rotated. The algorithm pieces them back together without any prior knowledge of the original image, using a combination of **Classical Computer Vision** (Color histograms, Sobel gradients) and **Deep Learning** (ResNet-18 embeddings).

## Installation
Ensure you have Python 3.8+ installed. Install the required dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage
To launch the GUI, run the demo application:

```bash
python demo_gui.py
```

From the GUI, you can:
1. Select a test image (e.g., `dog.jpg`, `ocean.jpg`, `acropolis.jpg`).
2. Choose the grid size (e.g., `4x4`, `8x8`, `16x16`).
3. Select the feature configuration (`Classical Only`, `Deep Only`, or `Combined`).
4. Click **Run Reconstruction** to view the step-by-step assembly and final evaluation metrics.

## Project Structure
- `M1_puzzle_generation.py`: Handles grid segmentation, random shuffling, and rotation tracking.
- `M2_3_feature_extraction.py`: Extracts color, texture, and pre-trained CNN (ResNet-18) features.
- `M4_adjacency_measures.py`: Calculates mathematical compatibility scores between piece boundaries.
- `M5_global_reconstruction.py`: Contains the greedy solver, simulated annealing optimizer, and global alignment logic.
- `M6_evaluation.py`: Computes quantitative metrics against the known ground truth.
- `demo_gui.py`: Handles the interactive PySimpleGUI application.
