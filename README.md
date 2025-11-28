# Hogwarts House Classification

A logistic regression implementation from scratch for multi-class classification of Hogwarts students into their houses.

## Mathematical Foundation

This project implements **multinomial logistic regression** using the One-vs-All (OvA) strategy:

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Hypothesis:**
```
h_θ(x) = σ(θ^T x)
```

**Gradient:**
```
∇J(θ) = (1/m) X^T (h_θ(x) - y)
```

**Parameter Update:**
```
θ := θ - α∇J(θ)
```

Where:
- `m` is the number of training examples
- `α` is the learning rate (0.001)
- `θ` are the model parameters (weights)

## Project Structure

- **model/log_train.py** - Training script with gradient descent implementation
- **model/predict.py** - Prediction script for house classification
- **data_analysis/describe.py** - Statistical analysis tool (reimplementation of pandas.describe())

## Usage

### Training
```bash
python model/log_train.py dataset_train.csv
```
Generates `weights.json` containing trained parameters for each house.

### Prediction
```bash
python model/predict.py dataset_test.csv weights.json
```
Outputs `houses.csv` with predicted Hogwarts houses.

### Data Analysis
```bash
python data_analysis/describe.py dataset_train.csv
```
Displays statistical summary (count, mean, std, quartiles) for numerical features.

## Features

- Custom implementations of mean, standard deviation, min, max, and quantiles
- Data preprocessing: normalization, encoding, missing value imputation
- Multi-class classification using One-vs-All strategy
- No scikit-learn for core model (only metrics for evaluation)
