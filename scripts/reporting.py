import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging 

from scripts.diagnostics import model_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

with open("config.json", "r") as f:
    config = json.load(f)

test_data_path = os.path.join(config["test_data_path"])
output_model_path = os.path.join(config["output_model_path"])


def save_confusion_matrix(test_data):
    logger.info('Calculate predictions on test data')
    preds = model_predictions(test_data)
    y_test = test_data["exited"].tolist()
    conf_mat = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(conf_mat, annot=True)
    ax.set_xlabel("predictions")
    ax.set_ylabel("target")
    plt.rc("font", size=50)
    fig.show()
    logger.info(f"Save confusion matrix to {output_model_path} folder")
    fig.savefig(output_model_path + "/confusionmatrix.png")

if __name__ == "__main__":
    test_data = pd.read_csv(test_data_path + "/testdata.csv")
    save_confusion_matrix(test_data)
