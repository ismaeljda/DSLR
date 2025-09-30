from log_train import logistic_regression, pre_processing
from sklearn.metrics import classification_report
import sys
import json
import pandas as pd
import numpy as np
encoder_house = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}

def predict(x, theta):
        z = np.dot(x, theta)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

def main():
    if len(sys.argv) != 3:
        print("Usage: python describe.py dataset_train.csv")
        return
    dataset = sys.argv[1]
    weights = sys.argv[2]
    try:
        data_train = pd.read_csv(dataset)
        X, data_train = pre_processing(data_train)
        with open(weights, "r") as file:
            weights = json.load(file) 
        all_preds = np.column_stack([predict(X, w) for w in weights.values()])
        final_pred = np.array(list(weights.keys()))[np.argmax(all_preds, axis=1)].astype(int)
        # Si on veut checker avec le train
        # Y = data_train['Hogwarts House']
        # print(classification_report(Y, final_pred))
        inverse_encoder = {v: k for k, v in encoder_house.items()}
        pred_str = [inverse_encoder[n] for n in final_pred]
        df_pred = pd.DataFrame({
        "Index": np.arange(len(final_pred)),
        "Hogwarts House": pred_str
        })
        df_pred.to_csv("houses.csv", index=False)
    except FileNotFoundError:
        print(f"Fichier introuvable : {dataset}")
    except pd.errors.EmptyDataError:
        print(f"Le fichier {dataset} est vide.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()