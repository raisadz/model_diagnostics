import logging
import pickle

from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def score_model(
    prod_deployment_path, output_model_path, test_data, dropped_columns=None
):
    logger.info(f"Load model from {prod_deployment_path}/trainedmodel.pkl")
    with open(prod_deployment_path + "/" + "trainedmodel.pkl", "rb") as f:
        model = pickle.load(f)
    if dropped_columns is not None:
        test_data = test_data.drop(columns=dropped_columns)
    X_test = test_data.copy()
    y_test = X_test.pop("exited")
    logger.info("Calculate prediction on test data")
    preds = model.predict(X_test)
    f1_test = f1_score(y_test, preds)
    logger.info(f"F1 score on test data is {f1_test}")
    with open(output_model_path + "/" + "latestscore.txt", "w") as f:
        f.write(str(round(f1_test, 6)))
    return round(f1_test, 6)
