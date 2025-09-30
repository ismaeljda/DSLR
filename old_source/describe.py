import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def count(df, feature):
    count = len(df[feature]) - df[feature].isna().sum()
    return count

def mean(df, feature):
    somme = df[feature].sum()
    mean = somme/count(df, feature)
    return mean

def std(df, feature):
    means = mean(df, feature)
    array_value = np.array(df[feature].dropna())
    variance = (((array_value - means)**2).sum())/count(df, feature)
    std = np.sqrt(variance)
    return std

def min(df, feature):
    df_fake = df.copy()
    df_fake = df_fake[df_fake[feature].notna()]
    df_fake.sort_values(by=feature, inplace = True)
    return df_fake[feature].iloc[0]

def max(df, feature):
    df_fake = df.copy()
    df_fake = df_fake[df_fake[feature].notna()]
    df_fake.sort_values(by=feature, inplace = True, ascending = False)
    return df_fake[feature].iloc[0]

def q(df, feature, percs):
    df_fake = df.copy()
    df_fake = df_fake[df_fake[feature].notna()]
    df_fake.sort_values(by=feature, inplace = True)
    nb_elem = count(df, feature)
    pos = int(percs * nb_elem)
    if nb_elem % 2 == 0:
        return (df_fake[feature].iloc[pos] + df_fake[feature].iloc[pos + 1])/2
    else:
        return (df_fake[feature].iloc[pos])
    
def describe(df):
    features = df.columns
    desc = {}
    for feature in features:
        desc[feature] = {}
        desc[feature]['count'] = count(df, feature)
        desc[feature]['mean'] = mean(df, feature)
        desc[feature]['std'] = std(df, feature)
        desc[feature]['Min'] = min(df, feature)
        desc[feature]['25%'] = q(df, feature, 0.25)
        desc[feature]['50%'] = q(df, feature, 0.5)
        desc[feature]['75%'] = q(df, feature, 0.75)
        desc[feature]['Max'] = max(df, feature)
    df_stats = pd.DataFrame.from_dict(desc, orient='index').T    
    return df_stats

def main():
    if len(sys.argv) != 2:
        print("Usage: python describe.py dataset_train.csv")
        return
    fichier = sys.argv[1]
    try:
        data_train = pd.read_csv(fichier)
        features_numeriques = data_train.select_dtypes(include=['int64', 'float64'])
        if 'Index' in features_numeriques.columns:
            features_numeriques = features_numeriques.drop('Index', axis=1)
        stats = describe(features_numeriques)
        print(stats)
    except FileNotFoundError:
        print(f"Fichier introuvable : {fichier}")
    except pd.errors.EmptyDataError:
        print(f"Le fichier {fichier} est vide.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()