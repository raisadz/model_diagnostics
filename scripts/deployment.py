import json
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
output_model_path = os.path.join(config["output_model_path"])

os.makedirs(prod_deployment_path, exist_ok=True)


def copy_to_deployment():
    logger.info(
        f"Copy trainedmodel.pkl from {output_model_path} folder "
        f"to {prod_deployment_path} folder"
    )
    shutil.copy2(f"{output_model_path}/trainedmodel.pkl", prod_deployment_path)
    logger.info(
        f"Copy latestscore.txt from {output_model_path} folder "
        f"to {prod_deployment_path} folder"
    )
    shutil.copy2(f"{output_model_path}/latestscore.txt", prod_deployment_path)
    logger.info(
        f"Copy ingestedfiles.txt from {dataset_csv_path} folder "
        f"to {prod_deployment_path} folder"
    )
    shutil.copy2(f"{dataset_csv_path}/ingestedfiles.txt", prod_deployment_path)


if __name__ == "__main__":
    copy_to_deployment()
