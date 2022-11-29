import json
import os
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def train_model(dataset_csv_path, model_path, dropped_columns = None):
    logger.info(f"Read datasets from {dataset_csv_path} folder")
    df = pd.read_csv(dataset_csv_path + "/" + "finaldata.csv")
    if dropped_columns is not None:
        df = df.drop(columns = dropped_columns)
    X = df.copy()
    y = X.pop('exited')

    lr = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="ovr",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    logger.info('Training logistic regression')
    model = lr.fit(X_train, y_train)
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    f1_train = f1_score(y_train, preds_train)
    f1_test = f1_score(y_test, preds_test)
    logger.info(f'F1 score on training is {f1_train}')
    logger.info(f'F1 score on test is {f1_test}')
    logger.info(f'Saving trained model to {model_path}/trainedmodel.pkl')
    with open(model_path + "/" + "trainedmodel.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config["output_folder_path"])
    model_path = os.path.join(config["output_model_path"])

    os.makedirs(model_path, exist_ok=True)

    train_model(dataset_csv_path=dataset_csv_path, model_path=model_path, dropped_columns= ['corporation'])
