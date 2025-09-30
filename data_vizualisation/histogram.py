import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

def histogram(data):
    houses = data['Hogwarts House'].unique()
    features_numeriques = data.select_dtypes(include=['int64', 'float64']).drop('Index', axis = 1)
    features = features_numeriques.columns
    features_numeriques['houses'] = data['Hogwarts House']
    nb_line = math.ceil(len(features) / 5)
    fig, axs = plt.subplots(nb_line, 5, figsize=(15, 3 * nb_line))
    axs = axs.flatten()

    for i, feature in enumerate(features):
        if i < len(axs):
            for house in houses:
                subset = features_numeriques[features_numeriques['houses'] == house][feature].dropna()
                axs[i].hist(subset, bins=30, alpha=0.5, label=house, edgecolor='black')
            
            axs[i].set_title(feature)
            axs[i].legend()

    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python describe.py dataset_train.csv")
        return
    fichier = sys.argv[1]
    try:
        data_train = pd.read_csv(fichier)
        histogram(data_train)
    except FileNotFoundError:
        print(f"Fichier introuvable : {fichier}")
    except pd.errors.EmptyDataError:
        print(f"Le fichier {fichier} est vide.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()