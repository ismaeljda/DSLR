import pandas as pd
import numpy as np
import sys
import os 

def ft_count(df, feature):
    count = len(df[feature]) - df[feature].isna().sum()
    return count
def ft_mean(df, feature):
    somme = df[feature].sum()
    mean = somme / ft_count(df, feature)
    return mean
def ft_std(df, feature):
    mean = ft_mean(df, feature)
    count = ft_count(df, feature)
    variance = ((df[feature] - mean)**2).sum()/ count
    std = np.sqrt(variance)
    return std
def ft_min(df, feature):
    df_fake = df.copy()
    df_fake = df_fake[df_fake[feature].notna()]
    df_fake.sort_values(by=feature, inplace = True)
    return df_fake[feature].iloc[0]
def ft_max(df, feature):
    df_fake = df.copy()
    df_fake = df_fake[df_fake[feature].notna()]
    df_fake.sort_values(by=feature, inplace = True, ascending= False)
    return df_fake[feature].iloc[0]
def ft_q(df, feature, num):
    df_fake = df.copy()
    df_fake = df_fake[df_fake[feature].notna()]
    df_fake.sort_values(by=feature, inplace = True)
    nb_elem = ft_count(df, feature)
    pos = int(nb_elem * num)
    if nb_elem % 2 == 0:
        return (df_fake[feature].iloc[pos] + df_fake[feature].iloc[pos + 1]) / 2
    else :
        return df_fake[feature].iloc[pos]
def describe(df):
    features = df.columns
    desc = {}
    for feature in features:
        desc[feature] = {}
        desc[feature]['count'] = ft_count(df, feature)
        desc[feature]['mean'] = ft_mean(df, feature)
        desc[feature]['std'] = ft_std(df, feature)
        desc[feature]['min'] = ft_min(df, feature)
        desc[feature]['25%'] = ft_q(df, feature, 0.25)
        desc[feature]['50%'] = ft_q(df, feature, 0.5)
        desc[feature]['75%'] = ft_q(df, feature, 0.75)
        desc[feature]['max'] = ft_max(df, feature)
    stat_df = pd.DataFrame.from_dict(desc, orient='index').T
    return stat_df

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