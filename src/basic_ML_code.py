import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow, mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.01)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.01)
args = parser.parse_args()

#evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from local
    data = pd.read_csv("data/red-wine-quality.csv")
    data.to_csv("data/red-wine-quality.csv", index=False)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    exp = mlflow.set_experiment(experiment_name="Frist Experiment v2")
    # exp_id = mlflow.create_experiment(name="Frist Create Experiment")
    # get_exp = mlflow.get_experiment(exp_id)

    with mlflow.start_run(experiment_id=exp.experiment_id, log_system_metrics=True):
    # with mlflow.start_run(experiment_id=exp_id):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        params = {
            "Alpha": alpha,
            "l1_ratio": l1_ratio
        }
        mlflow.log_params(params)

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
        mlflow.log_metrics(metrics)
        # mlflow.log_param("Alpha", alpha)
        # mlflow.log_param("l1_ratio", l1_ratio)
        # mlflow.log_metric("RMSE", rmse)
        # mlflow.log_metric("MAE", mae)
        # mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(lr, "myModel")

