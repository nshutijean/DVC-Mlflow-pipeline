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

# logging warning messages
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)