# Hogwarts House Classification

A logistic regression implementation from scratch for multi-class classification of Hogwarts students into their houses.

## Mathematical Foundation

### Logistic Regression

Logistic regression models the probability that an instance belongs to a particular class using the sigmoid function.

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Hypothesis:**
```
h_θ(x) = σ(θ^T x)
where z = θ^T x = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

**Cost Function (Log Loss):**
```
J(θ) = -(1/m) Σ[y·log(h_θ(x)) + (1-y)·log(1-h_θ(x))]
```

**Gradient of Cost Function:**
```
∇J(θ) = (1/m) X^T (h_θ(x) - y)
```

**Gradient Descent Update:**
```
θ := θ - α∇J(θ)
```

### One-vs-All Strategy

For multi-class classification (4 Hogwarts houses), we train **4 separate binary classifiers**:
- Each classifier learns to distinguish one house from all others
- For prediction, we run all 4 classifiers and select the house with highest probability:
```
predicted_house = argmax(h_θ^(i)(x)) for i ∈ {Gryffindor, Slytherin, Ravenclaw, Hufflepuff}
```

Where:
- `m` is the number of training examples
- `α` is the learning rate (0.001)
- `θ` are the model parameters (weights)
- `y ∈ {0,1}` for binary classification per house

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
