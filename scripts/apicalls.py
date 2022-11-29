import json
import logging
import os

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

URL = "http://127.0.0.1:8000"

with open("config.json", "r") as f:
    config = json.load(f)
output_model_path = os.path.join(config["output_model_path"])

logger.info("Prediction endpoint")
response_prediction = requests.get(
    f"{URL}/prediction?filename=testdata/testdata.csv"
).content
logger.info("Scoring endpoint")
response_scoring = requests.get(f"{URL}/scoring").content
logger.info("Summarystats endpoint")
response_summarystats = requests.get(f"{URL}/summarystats").content
logger.info("Diagnostics endpoint")
response_diagnostics = requests.get(f"{URL}/diagnostics").content

responses = (
    response_prediction.decode()
    + "\n"
    + response_scoring.decode()
    + "\n"
    + response_summarystats.decode()
    + "\n"
    + response_diagnostics.decode()
)

logger.info(f"Write all responses to {output_model_path}/apireturns.txt")
with open(output_model_path + "/apireturns.txt", "w") as f:
    f.write(responses)
