import json
import os

import pandas as pd
from flask import Flask, request, jsonify, make_response

from scripts.diagnostics import (
    dataframe_percent_na,
    dataframe_summary,
    execution_time,
    model_predictions,
    outdated_packages_list,
)
from scripts.scoring import score_model

app = Flask(__name__)
app.config.from_pyfile(os.path.join('../settings.py'))

with open("config.json", "r") as f:
    config = json.load(f)

test_data_path = os.path.join(config["test_data_path"])
output_folder_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
output_model_path = os.path.join(config["output_model_path"])

@app.route("/prediction", methods=["GET", "OPTIONS"])
def prediction():
    filename = request.args.get("filename")
    thedata = pd.read_csv(filename)
    preds = model_predictions(thedata)
    return make_response(jsonify(preds), 200) 


@app.route("/scoring", methods=["GET", "OPTIONS"])
def scoring():
    filename = test_data_path + "/testdata.csv"
    thedata = pd.read_csv(filename)
    f1_test = score_model(prod_deployment_path, output_model_path, thedata, dropped_columns = ['corporation'])
    return make_response(jsonify(f1_test), 200)


@app.route("/summarystats", methods=["GET", "OPTIONS"])
def summarystats():
    filename = output_folder_path + "/finaldata.csv"
    thedata = pd.read_csv(filename)
    df_summary = dataframe_summary(thedata)
    return make_response(jsonify(df_summary), 200) 


@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics():
    filename = output_folder_path + "/finaldata.csv"
    thedata = pd.read_csv(filename)
    return_time = execution_time()
    na_percent = dataframe_percent_na(thedata)
    outdated_req = outdated_packages_list()
    diagnostics_response = str(return_time) + '\n' + str(na_percent) + '\n' + str(outdated_req)
    return diagnostics_response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
