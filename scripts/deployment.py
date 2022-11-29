import json
import os

# Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
output_model_path = os.path.join(config["output_model_path"])

os.makedirs(prod_deployment_path, exist_ok=True)


# function for deployment
def copy_to_deployment():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    os.system(f"cp {output_model_path}/trainedmodel.pkl {prod_deployment_path}")
    os.system(f"cp {output_model_path}/latestscore.txt {prod_deployment_path}")
    os.system(f"cp {dataset_csv_path}/ingestedfiles.txt {prod_deployment_path}")


if __name__ == "__main__":
    copy_to_deployment()
