# Burn Classifier Model

This model is designed for multi-class classification to classify different types of burns in images.

## Overview

The burn classifier model uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It processes image data to classify burn images into multiple categories (e.g., different burn degrees or types).

## Prerequisites

- Python 3.8
- TensorFlow 2.6 with GPU support
- Other dependencies listed in `../../environment-full.yml`

To set up the environment:
```
conda env create -f ../../environment-full.yml
conda activate tensorflow-gpu
```

## Data Preparation

The dataset consists of images of different burn types. The data is not publicly available due to privacy and ethics constraints.

To prepare the data:
- Place images in directories for each class under an input directory.
- The expected class directory names are:
  - `Partial_thickness_burn`
  - `Normal_skin`
  - `Full_thickness_burn`
- Run: `python newdat.py <input_directory> <output_directory>`

This script:
- Loads images from the input directory.
- Applies augmentations and rotation for training images.
- Splits into train/test sets and shuffles the data.
- Saves processed files into `<output_directory>/data/`, including `x.pickle`, `y.pickle`, `x_test.pickle`, `y_test.pickle`, and one-hot label files.

If you have a SLURM cluster environment, use `data.sh` which submits the job:
```
sbatch data.sh
```
(Note: Adjust paths in `data.sh` for your cluster.)

## Hyperparameter Optimization

To find optimal model hyperparameters:
- Run: `python model.py <project_directory>`

This script:
- Loads the prepared data.
- Performs Bayesian optimization.
- Searches for best architecture parameters.
- Saves the best parameters to `best_params.pickle`.

## Training the Model

To train the model with optimized parameters:
- Run: `python train.py <project_directory>`

This script:
- Loads `best_params.pickle`.
- Builds the CNN model for multi-class classification.
- Trains on the data with early stopping and TensorBoard logging.
- Saves the trained model to `burn_multiple_class.model/`.

## Evaluation

To evaluate the trained model:
- Run: `python evaluation.py <project_directory>`

This script:
- Loads the model and test data.
- Computes accuracy, loss, and classification metrics.
- Saves evaluation results.

For SLURM: `sbatch evaluation.sh`

## Using the Trained Model for Prediction

If you already have a trained model in `burn_multiple_class.model/`, you can load it and use it to classify new burn images.

Example usage for one image:
```python
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('burn_multiple_class.model')
img = cv2.imread('path/to/image.jpg')
img = cv2.resize(img, (100, 100))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
class_labels = ['Partial_thickness_burn', 'Normal_skin', 'Full_thickness_burn']
class_idx = np.argmax(prediction, axis=1)[0]
label = class_labels[class_idx]
print('Predicted label:', label)
print('Scores (confidence):', prediction)
```

You can turn this into a prediction API endpoint with Flask or FastAPI. A simple endpoint should:
- accept an image file
- preprocess it to the model input shape
- call `model.predict()`
- return the predicted class label and confidence

## Additional Scripts

- `GPU.py`: Checks GPU availability.
- `best.py`: Possibly selects the best model.
- `tensorflow-gpu.sh`: SLURM script to load TensorFlow GPU module.

## Recreating the Model

1. Set up the conda environment as above.
2. Prepare your dataset in the required format.
3. Run data preparation: `python newdat.py <input> <output>` (adjust script name if needed)
4. Optimize hyperparameters: `python model.py <output>`
5. Train: `python train.py <output>`
6. Evaluate: `python evaluation.py <output>`

## Notes

- All scripts assume the project directory structure.
- Adjust file paths if necessary.
- For SLURM scripts, modify account, partition, etc., for your cluster.</content>
<parameter name="filePath">/workspaces/burn-ml/burn_clasifier_model/README.md