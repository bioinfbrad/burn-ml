[![DOI](https://zenodo.org/badge/1037282235.svg)](https://doi.org/10.5281/zenodo.19708992)


# Burn-ML
This repository contains two separate deep learning projects for burn image analysis.

- `burn_detiction_model/`: a binary burn detection model that separates burned skin from normal skin.
- `burn_clasifier_model/`: a multi-class burn classifier that distinguishes among burn types.

Each model directory includes a dedicated `README.md` with setup, data preparation, training, evaluation, and prediction guidance.

Use the top-level `environment-full.yml` to create the required Conda environment.

This repository contains machine learning models for burn detection and classification using image data.

## Overview

The repository is organized into two main model directories:

### burn_detiction_model

A binary classification model designed to detect whether an image contains burned skin or normal healthy skin. It includes scripts for data preparation, hyperparameter optimization, training, and evaluation of a CNN model.

### burn_clasifier_model

A multi-class classification model for classifying different types of burns (e.g., various burn degrees). Similar to the detection model, it provides data processing, optimization, training, and evaluation scripts for a CNN classifier.

## Getting Started

1. Set up the environment using the provided `environment-full.yml`:
   ```
   conda env create -f environment-full.yml
   conda activate tensorflow-gpu
   ```

2. Refer to the README in each model directory for detailed instructions on data preparation, training, and evaluation.

## Notes

- Datasets are not publicly available due to ethical and privacy constraints.
- Scripts include options for SLURM cluster environments.
- All models use TensorFlow/Keras for deep learning.
