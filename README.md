# Multi-label Classification with Image and Text

This project implements a multi-label classification model that combines image and text data using a custom CNN and text embedding models. The model is trained using PyTorch and handles class imbalance, data augmentation, and evaluation metrics for multi-label classification.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)

## Prerequisites

Ensure you have the following software installed:
- Python 3.7 or higher
- PyTorch
- torchvision
- scikit-learn
- pandas
- numpy
- torchmetrics
- torchtext
- PIL (Pillow)



- `train.csv` and `test.csv`: CSV files containing training and testing data with columns 'ImageID', 'Caption', and 'Labels'.
- `fixed_data/`: Directory containing the processed images.
- `vocab.pkl`: The vocabulary file generated from the captions.
- `main.py`: The main script containing the code for data preprocessing, model definition, training, and evaluation.
- `README.md`: This file.

## Setup

1. **Clone the repository:**
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install the required Python packages:**
    ```sh
    pip install torch torchvision scikit-learn pandas numpy torchmetrics torchtext pillow
    ```

3. **Prepare the dataset:**
    - Ensure `train.csv` and `test.csv` are in the project root directory.
    - Ensure the images mentioned in the CSV files are placed in the `fixed_data/` directory.

## Usage

1. **Preprocess the images:**
    ```python
    from main import process_dataset
    process_dataset('path/to/input_folder', 'path/to/output_folder', target_size=300)
    ```

2. **Build the vocabulary:**
    ```python
    from main import build_vocab
    build_vocab('train.csv', 'test.csv', 'vocab.pkl')
    ```

3. **Train the model:**
    ```python
    from main import train_model
    train_model('train.csv', 'vocab.pkl')
    ```

## Training

To start training the model, run the following command:
```sh
python main.py

