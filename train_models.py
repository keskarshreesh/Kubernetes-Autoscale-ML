from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import argparse
import os
from dotenv import load_dotenv
import joblib

load_dotenv()

MODELS_DIR = "../models"
DATA_DIR = "../data/filtered"


def train_linear_regression(data_dir="../data/filtered"):
    data = pd.read_csv(f"{data_dir}/filtered.csv")
    X = data.drop(columns=["average rps", "median latency"])
    y = data[["average rps", "median latency"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Linear Regression Mean Squared Error: {mse}")
    return lin_reg


def train_mlp(data_dir="../data/filtered"):
    data = pd.read_csv(f"{data_dir}/filtered.csv")
    X = data.drop(columns=["average rps", "median latency"])
    y = data[["average rps", "median latency"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MLP Mean Squared Error: {mse}")
    return mlp

def evaluate_linear_regression_model(model, data):
    X = data.iloc[:, :-1]
    y_true = data.iloc[:, -1]
    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)

    return {"mse": mse}


def evaluate_mlp_model(model, data):
    X = data.iloc[:, :-1]
    y_true = data.iloc[:, -1]
    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)
    return {"mse": mse}

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate linear regression and MLP models.")
    parser.add_argument("--mode", type=str, default="train", help="Train or evaluate mode.")
    args = parser.parse_args()

    if args.mode not in ["train", "evaluate"]:
        print("Invalid mode specified. Please choose 'train' or 'evaluate'.")
        return

    model_type = os.getenv("MODEL", "lr")

    if model_type == "lr":
        model_file = os.path.join(MODELS_DIR, "linear_regression_model.joblib")
        model_fn = train_linear_regression
        evaluate_fn = evaluate_linear_regression_model
    elif model_type == "mlp":
        model_file = os.path.join(MODELS_DIR, "mlp_model.joblib")
        model_fn = train_mlp
        evaluate_fn = evaluate_mlp_model
    else:
        print("Invalid model type specified. Please choose 'lr' or 'mlp'.")
        return

    if args.mode == "train":
        data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        model = model_fn(data)
        joblib.dump(model, model_file)
        print(f"Model saved to {model_file}.")
    else:
        data = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        model = joblib.load(model_file)
        results = evaluate_fn(model, data)
        print(f"Mean Squared Error: {results['mse']}")
        print(f"R-squared: {results['r2']}")


if __name__ == "__main__":
    main()






