import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow, mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--penalty", type=str, required=False, default="l2")
parser.add_argument("--solver", type=str, required=False, default="lbfgs")
args = parser.parse_args()

#evaluation function
def eval_metrics(actual, pred):
    accuracy_scor = accuracy_score(actual, pred)
    f1_scor = f1_score(actual, pred, average='macro')
    return accuracy_scor, f1_scor

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from local
    data = pd.read_csv("data/iris.csv")
    data.to_csv("data/iris.csv", index=False)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["variety"], axis=1)
    test_x = test.drop(["variety"], axis=1)
    train_y = train[["variety"]]
    test_y = test[["variety"]]
    train_x.to_csv("data/train_x_iris.csv", index=False)
    test_x.to_csv("data/test_x_iris.csv", index=False)
    train_y.to_csv("data/train_y_iris.csv", index=False)
    test_y.to_csv("data/test_y_iris.csv", index=False)
    

    penalty = args.penalty
    solver = args.solver

    exp = mlflow.set_experiment(experiment_name="IRIS Experiment")
    with mlflow.start_run(experiment_id=exp.experiment_id, log_system_metrics=True):
    # with mlflow.start_run(experiment_id=exp_id):
        lr = LogisticRegression(penalty=penalty, solver=solver, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (accuracy_scor, f1_scor) = eval_metrics(test_y, predicted_qualities)

        tags = {
            "engineering": "ML Platform",
            "release.candidate": "RC1",
            "release.version": "2.2.0",
            "release.data": "IRIS",
            "release.coordinator": "DhanaSekhar Buddha",
        }

        mlflow.set_tags(tags)

        print("  accuracy_score: %s" % accuracy_scor)
        print("  f1_score: %s" % f1_scor)

        params = {
            "penalty": penalty,
            "solver": solver
        }
        mlflow.log_params(params)

        metrics = {
            "accuracy_score": accuracy_scor,
            "f1_score": f1_scor
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr, "IRISModel")
        mlflow.log_artifacts("data/")
    print(mlflow.last_active_run())


