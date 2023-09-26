import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import category_encoders as ce

import warnings
import sys
import logging

import mlflow
import mlflow.sklearn
import dvc.api

# log warning messages
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get data url from DVC
data_url = dvc.api.get_url(
    path='data/car_evaluation_processed.csv',
    repo= '.',
    version = 'v1'
)

# define metrics for model perfromance evaluatio
def model_eval(actual, pred):
    accuracy = accuracy_score(actual, pred)
    return accuracy