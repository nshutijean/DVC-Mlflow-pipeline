import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import category_encoders as ce

import warnings
import sys
import os
import logging

import mlflow
import mlflow.sklearn
import dvc.api

# log warning messages
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get data url from DVC
path = 'data/car_evaluation_processed.csv'
repo = '.'
version = 'v1'

data_url = dvc.api.get_url(
    path=path,
    repo=repo,
    rev = version
)

# define metrics for model perfromance evaluation
def model_eval(actual, pred):
    """
    Calculating the accuracy score between actual and predicted values

    Inp: actual, predicated values
    Out: accuracy score
    """
    accuracy = accuracy_score(actual, pred)

    return accuracy

# set an experiment name
mlflow.set_experiment("car-evaluation")

# log parameters
def log_data_params(data_url, data_version, data):
    """
    Logging data parameters to MLflow

    Inp: any
    Out: none
    """
    mlflow.log_param("data_url", data_url)
    mlflow.log_param("data_version", data_version)
    mlflow.log_param("num_rows", data.shape[0])
    mlflow.log_param("num_cols", data.shape[1])

# encode features
def encode_features(X_train, X_test, columns):
    """
    Encoding the features with ordinal encoder

    Inp: X_train, X_test, columns
    Out: X_train_encoded, X_test_encoded
    """

    oe = ce.OrdinalEncoder(cols=columns)
    X_train_encoded = oe.fit_transform(X_train)
    X_test_encoded = oe.transform(X_test)

    return X_train_encoded, X_test_encoded

# execute the training pipeline
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # read data from the remote repository
    data = pd.read_csv(data_url, sep=",")

    # initialize mlflow
    with mlflow.start_run():

        # log data parameters
        log_data_params(data_url, version, data)

        # X and y split
        X = data.drop(['class'], axis=1)
        y = data['class']

        # split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # create an 'artifacts' directory
        directory = "./artifacts"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # log artifacts: features
        X_cols = pd.DataFrame(list(X_train.columns))
        X_cols.to_csv('artifacts/features.csv', header=False, index=False)
        mlflow.log_artifact('artifacts/features.csv')

        # log artifacts: targets
        y_cols = pd.DataFrame(list(y_train.unique())) 
        y_cols.to_csv('artifacts/targets.csv', header=False, index=False)
        mlflow.log_artifact('artifacts/targets.csv')

        # encode the features
        cols = ['buying', 'meant', 'doors', 'persons', 'lug_boot', 'safety']
        X_train, X_test = encode_features(X_train, X_test, cols)

        # set model parameters
        criterion = sys.argv[0] if len(sys.argv) > 1 else "entropy"
        max_depth = float(sys.argv[1]) if len(sys.argv) > 2 else 3

        # training the model
        dtc = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
        dtc.fit(X_train, y_train)

        # predicted values
        y_pred = dtc.predict(X_test)

        # model performance using accuracy
        accuracy = model_eval(y_test, y_pred)

        # log the metric
        mlflow.log_metric("accuracy", accuracy)

        # print("Accuracy:, accuracy)

    mlflow.end_run()