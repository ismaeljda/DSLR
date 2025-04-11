import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

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

def gradient_descent(x, y, theta, learning_rate, iterations=1000):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        h = predict(x, theta)
        gradient = (1/m) * np.dot(x.T, (h - y))
        theta -= learning_rate * gradient
        cost_history.append(cost_loss(x,y, theta))
    return cost_history, theta

def one_vs_all_train(X, y, num_classes, alpha=0.1, num_iter=1000):
    m, n = X.shape
    all_theta = np.zeros((num_classes, X.shape[1]))
    costs = []
    for i in range(num_classes):
        y_binary = np.where(y == i, 1, 0)
        theta = np.zeros(n)
        cost_history, theta = gradient_descent(X, y_binary, theta, alpha, num_iter)
        costs.append(cost_history)
        all_theta[i] = theta
        
    return all_theta, costs

def one_vs_all_predict(X, all_theta):
    probs = sigmoid(np.dot(X, all_theta.T))
    return np.argmax(probs, axis=1)
import matplotlib.pyplot as plt

def plot_cost(costs_list, labels=None):
    """
    Affiche les graphes représentant le coût par itération pour chaque modèle.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, costs in enumerate(costs_list):
        if i < len(axes):
            axes[i].plot(range(len(costs)), costs, color='blue')
            axes[i].set_xlabel("Itérations (x100)")
            axes[i].set_ylabel("Coût")
            
            # Correction de la ligne problématique
            if labels is not None and i < len(labels):
                title = labels[i]
            else:
                title = f"Classe {i}"
                
            axes[i].set_title(title)
            axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

# def main():
#     data_train = pd.read_csv('datasets/dataset_train.csv')
#     features_numeriques = data_train.select_dtypes(include=['int64', 'float64']).drop('Index', axis = 1)
#     features_numeriques['houses'] = data_train['Hogwarts House']
#     data = features_numeriques.copy().dropna()
#     y = data['houses']
#     x = data.drop('houses', axis=1)
#     x = np.c_[np.ones(x.shape[0]), x]
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#     label_encoder = LabelEncoder()
#     y_train_encoded = label_encoder.fit_transform(y_train)
#     y_test_encoded = label_encoder.fit_transform(y_test)
#     num_classes = len(np.unique(y_train_encoded))
#     theta_all, costs = one_vs_all_train(x_train, y_train_encoded, num_classes)
#     y_pred = one_vs_all_predict(x_test, theta_all)
#     y_tests = np.array(y_test_encoded)
#     print(classification_report(y_test_encoded, y_pred))

def main():
    # Charger et préparer les données
    print("Chargement des données...")
    data_train = pd.read_csv('datasets/dataset_train.csv')
    features_numeriques = data_train.select_dtypes(include=['int64', 'float64']).drop('Index', axis=1)
    features_numeriques['houses'] = data_train['Hogwarts House']
    data = features_numeriques.copy().dropna()
    
    # Encoder les maisons
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['houses'])
    X = data.drop('houses', axis=1).values
    
    # Normalisation des données - TRÈS IMPORTANT
    print("Normalisation des données...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Ajouter la colonne de biais
    X = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraîner le modèle one-vs-all
    print("Entraînement du modèle one-vs-all...")
    num_classes = len(np.unique(y))
    
    # Utiliser un learning rate plus faible et plus d'itérations
    all_theta, costs = one_vs_all_train(X_train, y_train, num_classes, alpha=0.01, num_iter=5000)
    
    # Prédictions avec notre modèle
    print("Prédictions avec notre modèle...")
    y_pred = one_vs_all_predict(X_test, all_theta)
    
    # Évaluation des performances
    print("\nRésultats avec notre implémentation:")
    print(classification_report(y_test, y_pred))

    print("Visualisation des courbes de coût...")
    house_names = label_encoder.classes_
    plot_cost(costs, house_names)

if __name__ == "__main__":
    main()
 