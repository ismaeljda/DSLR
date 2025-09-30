import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import seaborn as sns

def pair_plot(data):
    features_numeriques = data.select_dtypes(include=['int64', 'float64']).drop('Index', axis=1)
    
    features_numeriques['Hogwarts House'] = data['Hogwarts House']
    
    sns.pairplot(features_numeriques, hue='Hogwarts House', diag_kind='hist', corner=True, plot_kws={'alpha': 0.6})
    plt.suptitle("Pair Plot des features numériques", y=1.02)
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python describe.py dataset_train.csv")
        return
    fichier = sys.argv[1]
    try:
        data_train = pd.read_csv(fichier)
        pair_plot(data_train)
    except FileNotFoundError:
        print(f"Fichier introuvable : {fichier}")
    except pd.errors.EmptyDataError:
        print(f"Le fichier {fichier} est vide.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()