import io
import json
import logging
import os
import pickle
import subprocess
import timeit

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])


def model_predictions(input_data):
    logger.info("Running function model_predictions")
    logger.info(f"Reading model from {prod_deployment_path}/trainedmodel.pkl")
    with open(prod_deployment_path + "/" + "trainedmodel.pkl", "rb") as f:
        model = pickle.load(f)
    X_test = np.array(
        input_data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    )
    logger.info("Calculating predictions")
    preds = model.predict(X_test).tolist()
    return preds


def dataframe_summary(input_data):
    logger.info("Running function dataframe_summary")
    numeric_cols = list(input_data.select_dtypes("number").columns)
    stats = input_data[numeric_cols].agg(["mean", "median", "std"])
    statistics_list = stats.values.reshape(-1, 1).ravel().tolist()
    return statistics_list


def dataframe_percent_na(input_data):
    logger.info("Running function calculating the percent of NA")
    return (input_data.isna().sum() / input_data.shape[0]).tolist()


def execution_time():
    logger.info("Calculating ingestion execution time")
    start_ingestion_time = timeit.default_timer()
    os.system("python scripts/ingestion.py")
    ingestion_time = timeit.default_timer() - start_ingestion_time

    logger.info("Calculating training execution time")
    start_training_time = timeit.default_timer()
    os.system("python scripts/training.py")
    training_time = timeit.default_timer() - start_training_time

    return_time = [ingestion_time, training_time]
    return return_time


def outdated_packages_list():
    logger.info("Finding outdated packages")
    out = subprocess.run(
        ["pip", "list", "--outdated"], capture_output=True, text=True
    ).stdout
    out_requirements = pd.read_csv(io.StringIO(out), sep="\s+").iloc[:, :3]
    return out_requirements


if __name__ == "__main__":
    input_data = pd.read_csv(dataset_csv_path + "/" + "finaldata.csv")
    print(model_predictions(input_data))
    print(dataframe_summary(input_data))
    print(dataframe_percent_na(input_data))
    print(execution_time())
    print(outdated_packages_list())
