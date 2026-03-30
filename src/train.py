import argparse
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json


def main():
    parser = argparse.ArgumentParser(
        description="Train a Music Popularity Prediction model."
    )
    parser.add_argument("--n_estimators", type=int, default=145, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=20, help="Maximum depth")
    parser.add_argument(
        "--min_samples_split", type=int, default=9, help="Min samples split"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/prepared/",
        help="Path to the train dataset",
    )
    parser.add_argument(
        "--model_path", type=str, default="models/", help="Path to save model"
    )
    args = parser.parse_args()

    df_train = pd.read_csv(args.data_path + "train.csv")
    X_train = df_train.drop("popularity", axis=1)
    y_train = df_train["popularity"]

    df_test = pd.read_csv(args.data_path + "test.csv")
    X_test = df_test.drop("popularity", axis=1)
    y_test = df_test["popularity"]

    mlflow.set_experiment("CI_Music_Popularity")

    with mlflow.start_run():
        mlflow.set_tag("developer", "Kordan_Pavlo")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset_version", "1.1")

        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "random_state": 42,
        }

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        plt.figure(figsize=(10, 8))
        feat_importances = pd.Series(model.feature_importances_, index=X_test.columns)
        feat_importances.nlargest(15).sort_values().plot(kind="barh", color="skyblue")
        plt.title("Top 15 Feature Importances")
        plt.xlabel("Importance Score")
        plt.tight_layout()

        plot_path = "feature_importance.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        mlflow.sklearn.log_model(model, "random_forest_model")
        os.makedirs(args.model_path, exist_ok=True)
        filename = args.model_path + "random_forest_model.pkl"
        with open(filename, "wb") as file:
            pickle.dump(model, file)
        metrics = {"rmse": rmse, "r2": r2}
        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Run complete! RMSE: {rmse:.4f}, R2: {r2:.4f}")


if __name__ == "__main__":
    main()
