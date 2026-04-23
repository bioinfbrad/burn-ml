# Burn Detection Model

This model is designed for binary classification to detect burns in images, distinguishing between burned skin and normal healthy skin.

## Overview

The burn detection model uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It processes image data to classify whether an image contains burned skin or normal skin.

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

The dataset consists of images of burned and normal skin. The data is not publicly available due to privacy and ethics constraints.

To prepare the data:
- Place images in directories named `burned_skin` and `Normal_skin` under an input directory.
- Run: `python Data.py <input_directory> <output_directory>`

This script:
- Loads images from the input directory.
- Applies balanced augmentation and rotation to increase training data while avoiding bias (e.g., different ratios for each class to balance the dataset).
- Splits into train/test sets and shuffles the data.
- Saves processed files into `<output_directory>/data/`, including `x.pickle`, `y.pickle`, `x_test.pickle`, `y_test.pickle`, and one-hot label files.

If you have a SLURM cluster environment, use `data.sh` which submits the job:
```
sbatch data.sh
```
(Note: Adjust paths in `data.sh` for your cluster.)

## Hyperparameter Optimization

To find optimal model hyperparameters:
- Run: `python Model.py <project_directory>`

This script:
- Loads the prepared data.
- Performs Bayesian optimization using scikit-optimize.
- Searches for best number of layers, filters, regularization, etc.
- Saves the best parameters to `best_params.pickle`.

## Training the Model

To train the model with optimized parameters:
- Run: `python train.py <project_directory>`

This script:
- Loads `best_params.pickle`.
- Builds the CNN model.
- Trains on the data with early stopping.
- Saves the trained model to `Burn_images.model/`.

## Evaluation

To evaluate the trained model:
- Run: `python Evaluation.py <project_directory> <output_directory>`

This script:
- Loads the model and test data.
- Computes accuracy, loss, and other metrics.
- Saves evaluation outputs and plots into the given output directory.

For SLURM: `sbatch eval.sh`

## Using the Trained Model for Prediction

If you already have a trained model in `Burn_images.model/`, you can load it and use it to classify new images.

Example usage for one image:
```python
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('Burn_images.model')
img = cv2.imread('path/to/image.jpg')
img = cv2.resize(img, (75, 75))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
label_names = ['burned_skin', 'Normal_skin']
class_idx = np.argmax(prediction, axis=1)[0]
label = label_names[class_idx]
print('Predicted label:', label)
print('Scores:', prediction)
```

This can be wrapped in a simple API endpoint using Flask or FastAPI. For example, a prediction endpoint would:
- accept image uploads
- preprocess the image to the same size and scale
- call `model.predict()`
- return the predicted class and confidence scores

## Additional Scripts

- `GPU.py`: Checks and lists available GPUs.
- `plot.py`: Plots training/validation loss and accuracy from logs.
- `roc.py`: Generates and plots ROC curve for the model.
- `shape_extract.py`: Extracts shapes/features from images (utility).
- `burn_shape.py`: Related to burn shape analysis.
- `tensorflow-gpu.sh`: SLURM script to load TensorFlow GPU module.
- `extract_shape.sh`: SLURM job for shape extraction.
- `roc.sh`: SLURM job for ROC generation.

## Recreating the Model

1. Set up the conda environment as above.
2. Prepare your dataset in the required format.
3. Run data preparation: `python Data.py <input> <output>`
4. Optimize hyperparameters: `python Model.py <output>`
5. Train: `python train.py <output>`
6. Evaluate: `python Evaluation.py <output>`

## Notes

- All scripts assume the project directory structure.
- Adjust file paths if necessary.
- For SLURM scripts, modify account, partition, etc., for your cluster.</content>
<parameter name="filePath">/workspaces/burn-ml/burn_detiction_model/README.md