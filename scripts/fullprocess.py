import ast
import json
import logging
import os
import sys

import pandas as pd

from scripts.scoring import score_model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def check_new_data(input_folder_path):
    logging.info(f"Read data from {input_folder_path}")
    with open(prod_deployment_path + "/ingestedfiles.txt", "r") as f:
        ingestedfiles = ast.literal_eval(f.read())
    current_files = os.listdir(input_folder_path)
    new_files = [x for x in current_files if x not in ingestedfiles]
    return new_files


def check_model_drift(output_folder_path, prod_deployment_path, output_model_path):
    with open(prod_deployment_path + "/latestscore.txt", "r") as f:
        latestscore = ast.literal_eval(f.read())

    most_recent_data = pd.read_csv(output_folder_path + "/finaldata.csv")
    new_score = score_model(
        prod_deployment_path,
        output_model_path,
        most_recent_data,
        dropped_columns=["corporation"],
    )

    model_drift = new_score < latestscore
    return model_drift


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    prod_deployment_path = os.path.join(config["prod_deployment_path"])
    input_folder_path = os.path.join(config["input_folder_path"])
    output_folder_path = config["output_folder_path"]
    output_model_path = config["output_model_path"]

    new_files = check_new_data(input_folder_path)
    if new_files == []:
        logging.info("No new data found, exit the process")
        sys.exit()

    logging.info("Found new data, ingest it")
    os.system("python -m scripts.ingestion")

    model_drift = check_model_drift(
        output_folder_path, prod_deployment_path, output_model_path
    )
    if model_drift is False:
        logging.info("No model drift found, exit the process")
        sys.exit()
    else:
        logging.info("Model drift found, re-training and re-deployment")
        os.system("python -m scripts.training")
        os.system("python -m scripts.deployment")

    logging.info("Run diagnostics and reporting")
    os.system("python -m scripts.reporting")
    os.system("python -m scripts.app")
    os.system("python -m scripts.apicalls")
