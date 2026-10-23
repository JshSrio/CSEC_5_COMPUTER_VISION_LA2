# LA2: Can a Simple Neural Network Recognize Images?

**Dataset:** A Large Scale Fish Dataset (Kaggle)

## Overview
This Jupyter Notebook builds and evaluates a simple fully connected neural network that classifies fish images into categories based on their folder labels. It uses TensorFlow/Keras for the model and scikit-learn for splitting the data and evaluating classification performance.

The notebook explores the basic image-classification workflow: loading image files, preprocessing and normalizing them, splitting the data into training/validation/test sets, training a neural network, and evaluating its predictions.

## Objectives
- Load fish images from a local or Google Drive dataset directory.
- Resize images to 64 × 64 pixels and normalize pixel values to the range 0–1.
- Encode class labels for multi-class classification.
- Split the data into training, validation, and test sets using a stratified 70/15/15 split.
- Build and train a simple feed-forward neural network.
- Visualize training and validation accuracy and loss.
- Evaluate the trained model with test accuracy, a confusion matrix, and a classification report.

## Model Architecture
The notebook defines a sequential neural network with:
1. An input layer for 64 × 64 RGB images.
2. A `Flatten` layer to convert each image into a one-dimensional feature vector.
3. A dense layer with 256 units and ReLU activation.
4. A dropout layer with a rate of 0.3.
5. A dense layer with 128 units and ReLU activation.
6. A second dropout layer with a rate of 0.3.
7. A softmax output layer with one unit per class.

The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as its metric. Training is configured for 20 epochs with a batch size of 32.

## Technologies Used
- Python
- Jupyter Notebook / Google Colab
- TensorFlow and Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Requirements
Install the additional packages used in the notebook:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

TensorFlow installation options may differ by operating system and hardware. The notebook includes a cell to install `seaborn` and `scikit-learn`; TensorFlow and the other dependencies must also be available in the environment.

## Dataset Setup
The notebook is configured to use this Google Drive path:

```text
/content/drive/MyDrive/COMPUTER_VISON/archive/Fish_Dataset/Fish_Dataset
```

It also mounts Google Drive using `google.colab.drive`, so the notebook is currently designed for Google Colab.

To run it:
1. Upload or place the fish dataset in your Google Drive.
2. Confirm that the dataset directory contains image files organized in class-specific folders.
3. Update `DATASET_DIR` in the notebook if your directory differs.
4. Run the Drive-mounting cell and grant access.
5. Run the notebook cells in order.

The image loader accepts `.png`, `.jpg`, and `.jpeg` files. It skips directories whose names contain `GT`, which helps exclude ground-truth image folders. Labels are inferred from the name of the folder containing each image.

## Workflow
1. **Setup:** Import dependencies and set random seeds for Python, NumPy, and TensorFlow.
2. **Load image paths:** Recursively collect supported image files and use their parent folder names as labels.
3. **Preprocess:** Resize each image to 64 × 64, convert it to a numeric array, and divide pixel values by 255.
4. **Encode labels:** Convert folder labels to integer class IDs and one-hot encoded vectors.
5. **Split data:** Create stratified training (70%), validation (15%), and test (15%) sets.
6. **Build and compile:** Create the neural network and configure its optimizer, loss function, and metric.
7. **Train:** Fit the model for up to 20 epochs while monitoring validation performance.
8. **Visualize learning:** Plot training and validation accuracy and loss across epochs.
9. **Evaluate:** Measure test loss and accuracy, then display a confusion matrix and classification report.

## Generated Outputs
The notebook saves the following plots in its current working directory:

| File | Description |
|---|---|
| `training_curves.png` | Training and validation accuracy and loss across epochs |
| `confusion_matrix.png` | Heatmap comparing true class labels with predicted labels |

It also prints the test loss, test accuracy, and classification report (including precision, recall, and F1-score for each class) in the notebook output.

