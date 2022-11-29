import json
import os
import glob 
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]

os.makedirs(output_folder_path, exist_ok=True)

def merge_multiple_dataframe():
    logger.info(f"Read datasets from {input_folder_path} folder")
    datasets = glob.glob(f'{input_folder_path}/*.csv')
    logger.info("Merge datasets")
    df_total = pd.concat(map(pd.read_csv, datasets))
    logger.info("Clean data")
    df_dedup = clean_data(df_total)
    logger.info(f"write data to {output_folder_path}/finaldata.csv")
    df_dedup.to_csv(output_folder_path + "/finaldata.csv", index=False)
    with open(output_folder_path + "/ingestedfiles.txt", "w") as f:
        f.write(str(os.listdir()))

def clean_data(df):
    return df.drop_duplicates()


if __name__ == "__main__":
    merge_multiple_dataframe()
