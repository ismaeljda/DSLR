import os 
import sys
import json
import pandas as pd
import numpy as np

def pre_processing(data_train):
    data_train["Birthday"] = pd.to_datetime(data_train["Birthday"])
    data_train["year"] = data_train["Birthday"].dt.year
    data_train["month"] = data_train["Birthday"].dt.month
    data_train["day"] = data_train["Birthday"].dt.day
    data_train = data_train.drop(['Index','First Name','Last Name', 'Birthday'], axis=1)
    # encoding
    encoder_house = encoder(data_train, "Hogwarts House") #Dictionnaire avec mapping label et nom originel
    data_train['Hogwarts House'] = data_train['Hogwarts House'].map(encoder_house)
    encoder_hand = encoder(data_train, "Best Hand")
    data_train['Best Hand'] = data_train['Best Hand'].map(encoder_hand)
    fill_na(data_train)
    X = data_train.drop('Hogwarts House', axis=1)
    X_mean = X.mean()
    X_std = X.std()
    X = (X - X_mean) / X_std
    X = np.hstack([np.ones((X.shape[0], 1)), X]) #Rajouter la colone de 1 pour le parametre de biais 
    return X, data_train
def encoder(df, feature):
    encoder = {name: idx for idx, name in enumerate(df[feature].unique())}
    return encoder

def check_na(df):
    features = df.columns
    for feature in features:
        count = df[feature].isna().sum()
        print (f"{feature} has {count} missing value")

def fill_na(df):
    features = df.columns
    for feature in features:
        df[feature] = df[feature].fillna(df[feature].mean())

class logistic_regression:
    def __init__(self, x, y, learning_rate=0.001, max_iter=10000 ):
        self.x = x
        self.y = y
        self.theta = np.zeros(x.shape[1])
        self.m = len(y)
        self.learning_rate= learning_rate
        self.max_iter = max_iter
    def predict(self, x, theta):
        z = np.dot(x, theta)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    def gradient(self, y, theta):
        h = self.predict(self.x, theta)
        grad = 1/self.m * np.dot(self.x.T, (h - y))
        return grad
    def gradient_descent(self):
        house_gradient = {}
        houses = self.y.unique()
        for house in houses:
            y_new = (self.y == house).astype(int) #met 1 si true et 0 si false
            theta_house = self.theta.copy()
            for i in range(self.max_iter):
                gradient = self.gradient(y_new, theta_house)
                theta_house -= self.learning_rate * gradient
            house_gradient[str(house)]=theta_house.tolist()
        return house_gradient

def main():
    if len(sys.argv) != 2:
        print("Usage: python describe.py dataset_train.csv")
        return
    fichier = sys.argv[1]
    try:
        data_train = pd.read_csv(fichier)
        X,data_train = pre_processing(data_train)
        Y = data_train['Hogwarts House']
        model = logistic_regression(X,Y)
        weights = model.gradient_descent()
        with open("weights.json", "w") as file:
            json.dump(weights, file, indent=4) 
    except FileNotFoundError:
        print(f"Fichier introuvable : {fichier}")
    except pd.errors.EmptyDataError:
        print(f"Le fichier {fichier} est vide.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()