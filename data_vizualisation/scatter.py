import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import itertools

def scatter(data_train):
    features_numeriques = data_train.select_dtypes(include=['int64', 'float64']).drop('Index', axis = 1)
    features = features_numeriques.columns
    features_iter = list(itertools.combinations(features, 2))
    nb_line = math.ceil(len(features_iter) / 5)
    fig, axs = plt.subplots(nb_line, 5, figsize=(15, 3 * nb_line))
    axs = axs.flatten()

    for i, feature in enumerate(features_iter):
        if i < len(axs):
            axs[i].scatter(features_numeriques[feature[0]], features_numeriques[feature[1]])
            axs[i].set_title(f"{feature[0]} vs {feature[1]}")
            axs[i].set_xlabel(str(feature[0]))
            axs[i].set_ylabel(str(feature[1]))
    plt.tight_layout()
    plt.show()

#Methode pour differencier les differentes maison dans le nuage de points
def scatter_house(data_train):
    houses = data_train['Hogwarts House'].unique()
    features_numeriques = data_train.select_dtypes(include=['int64', 'float64']).drop('Index', axis = 1)
    features = features_numeriques.columns
    features_numeriques['houses'] = data_train['Hogwarts House']
    features_iter = list(itertools.combinations(features, 2))
    nb_line = math.ceil(len(features_iter) / 5)
    fig, axs = plt.subplots(nb_line, 5, figsize=(15, 3 * nb_line))
    axs = axs.flatten()

    for i, feature in enumerate(features_iter):
        if i < len(axs):
            for house in houses:
                subset = features_numeriques[features_numeriques['houses'] == house].dropna()
                axs[i].scatter(subset[feature[0]], subset[feature[1]], label = house)
                axs[i].set_title(f"{feature[0]} vs {feature[1]}")
                axs[i].set_xlabel(str(feature[0]))
                axs[i].set_ylabel(str(feature[1]))
                axs[i].legend()
    plt.tight_layout()
    plt.show()

#Methode pour afficher les graphs de manière plus lisibles
def scatter_graph(data_train):
    features_numeriques = data_train.select_dtypes(include=['int64', 'float64']).drop('Index', axis = 1)
    features = features_numeriques.columns
    features_iter = list(itertools.combinations(features, 2))
    nb_line = math.ceil(len(features_iter) / 5)
    step = 10
    for i in range(0, len(features_iter), step):
        sub_features = features_iter[i:i+step]
        nb_line = math.ceil(len(sub_features) / 4)
        fig, axs = plt.subplots(nb_line, 4, figsize=(15, 3 * nb_line))
        axs = axs.flatten()

        for j, feature in enumerate(sub_features):
            axs[j].scatter(features_numeriques[feature[0]], features_numeriques[feature[1]], alpha=0.6)
            axs[j].set_title(f"{feature[0]} vs {feature[1]}")
            axs[j].set_xlabel(feature[0])
            axs[j].set_ylabel(feature[1])
        
        for k in range(j+1, len(axs)):
            axs[k].axis('off')

        plt.tight_layout()
        plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python describe.py dataset_train.csv")
        return
    fichier = sys.argv[1]
    try:
        data_train = pd.read_csv(fichier)
        scatter_graph(data_train)
        # scatter_house(data_train)
    except FileNotFoundError:
        print(f"Fichier introuvable : {fichier}")
    except pd.errors.EmptyDataError:
        print(f"Le fichier {fichier} est vide.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()