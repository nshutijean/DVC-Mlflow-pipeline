## DVC and Mlflow pipeline

### Objective
The objective of this project is to demonstrate how DVC and Mlflow can be used together to version the data and track machine learning experiments.

### Data
We will use the Car Evaluation dataset from the UCI Machine Learning Repository for this demo. The dataset will be used for a classification task, where we will predict the evaluation of a car based on its attributes.

The target variable or class label is the evaluation of the car, which is categorized into four values: unacc (unacceptable), acc (acceptable), good, and vgood (very good).

It can be found here: https://archive.ics.uci.edu/dataset/19/car+evaluation

### Installation

To install the dependencies, you can use either pip or conda. Here are the steps:

Using pip:
```
python -m venv dvc-mlflow

source dvc-mlflow/bin/activate

pip install -r requirements.txt
```
Using conda:
```
conda create -n dvc-mlflow -y

conda install --yes --file requirements.txt
```

### Model
We utilized a DecisionTreeClassifier from scikit-learn to perform the classification task. The model was trained on the car evalution dataset and evaluated using accuracy as the metric.

### Usage
To run the workflow:

1. Clone this repo
2. Run `dvc init` to initialize DVC
3. Import the data using `dvc pull`
4. Execute the workflow and train the model using `python train.py` (this will also track the experiment with MLflow)
     - You can also use arguments like `python train.py gini 4` which signifies the criterion to use for splitting (gini) and the max depth of the tree (4).
5. Run `mlflow ui` to view the experiment in the MLflow UI