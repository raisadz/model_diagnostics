import json
import os

import pandas as pd

from scripts.utils import score_model

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    output_model_path = os.path.join(config["output_model_path"])
    test_data_path = os.path.join(config["test_data_path"])
    prod_deployment_path = os.path.join(config["prod_deployment_path"])
    test_data = pd.read_csv(test_data_path + "/testdata.csv")
    score_model(
        prod_deployment_path,
        output_model_path,
        test_data,
        dropped_columns=["corporation"],
    )
