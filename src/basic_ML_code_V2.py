import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Get arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.01)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.01)
args = parser.parse_args()

# Evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file
    data = pd.read_csv("data/red-wine-quality.csv")
    data.to_csv("data/red-wine-quality.csv", index=False)

    # Split the data into training and test sets (0.75, 0.25 split)
    train, test = train_test_split(data)

    # The predicted column is "quality", which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    exp = mlflow.set_experiment(experiment_name="First Experiment with signature")

    # Start the MLflow run
    with mlflow.start_run(experiment_id=exp.experiment_id, log_system_metrics=True):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("ElasticNet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
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

        # Infer the model signature based on the training data and the predicted output
        signature = infer_signature(train_x, predicted_qualities)

        # Log the model with the signature and an input example
        input_example = test_x.iloc[:1]  # Use a single test instance as input example
        mlflow.sklearn.log_model(lr, "myModel", signature=signature, input_example=input_example)
