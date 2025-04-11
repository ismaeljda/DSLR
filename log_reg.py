import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict(x, theta):
    z = np.dot(x, theta)
    return sigmoid(z)

def cost_loss(x, y, theta):
    m = len(y)
    h = predict(x,theta)
    h = np.clip(h, 1e-15, 1 - 1e-15)
    return (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient_descent(x, y, theta, learning_rate=0.01, iterations=1000):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        h = predict(x, theta)
        gradient = (1/m) * np.dot(x.T, (h - y))
        theta -= learning_rate * gradient
        cost_history.append(cost_loss(x,y, theta))
    return cost_history, theta

def main():
    data_train = pd.read_csv('datasets/dataset_train.csv')
    features_numeriques = data_train.select_dtypes(include=['int64', 'float64']).drop('Index', axis = 1)
    features_numeriques['houses'] = data_train['Hogwarts House']
    data = features_numeriques.copy().dropna()
    data['houses'] = data['houses'].apply(lambda x: 1 if x == 'Hufflepuff' else 0)
    y = data['houses']
    x = data.drop('houses', axis=1)
    x = np.c_[np.ones(x.shape[0]), x]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    theta = np.zeros(x_train.shape[1])
    cost_history, theta_final = gradient_descent(x_train,y_train, theta, learning_rate=0.1, iterations=1000)
    y_pred = predict(x_test, theta_final)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    y_tests = np.array(y_test)
    cm = confusion_matrix(y_tests, y_pred_binary)
    print(classification_report(y_tests, y_pred_binary))
    # print("Paramètres finaux : ", theta_final)

if __name__ == "__main__":
    main()
