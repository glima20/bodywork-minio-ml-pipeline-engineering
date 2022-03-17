from urllib.request import urlopen
from subprocess import CalledProcessError, run
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils import configure_logger, configure_mlflow

# MLFLOW_EXPERIMENT = "iris-classifier"
# MLFLOW_MODEL_NAME = f"{MLFLOW_EXPERIMENT}--sklearn-decision-tree"
DATA_URL = (
    "http://bodywork-ml-pipeline-project.s3.eu-west-2.amazonaws.com"
    "/data/iris_classification_data.csv"
)

log = configure_logger()


def main() -> None:
    """Main script to be executed."""
    try:
        configure_mlflow("mau")
        data = download_dataset(DATA_URL)
        log.info("data:{}".format(len(data)))
    except Exception as e:
        msg = f"training stage failed with exception: {e}"
        log.error(msg)
        raise RuntimeError(msg)


def download_dataset(url: str) -> pd.DataFrame:
    """Get data from cloud object storage."""
    log.info(f"Downloading training data from {DATA_URL}.")
    data_file = urlopen(url)
    return pd.read_csv(data_file)


if __name__ == "__main__":
    main()
