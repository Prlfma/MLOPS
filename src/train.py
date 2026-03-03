import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

def main():
    parser = argparse.ArgumentParser(description="Train a Music Popularity Prediction model.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum depth of the tree")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Min samples to split a node")
    parser.add_argument("--data_path", type=str, default="data/raw/dataset.csv", help="Path to the dataset")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    df.dropna(subset=['artists'], inplace=True)
    df = df[df.popularity != 0].copy()
    
    drop_cols = ["index", "track_id", "track_name", "album_name", "artists"]
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    
    df["explicit"] = df["explicit"].astype(int)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_genre = encoder.fit_transform(df[["track_genre"]])
    genre_columns = encoder.get_feature_names_out(["track_genre"])
    genre_df = pd.DataFrame(encoded_genre, columns=genre_columns, index=df.index)

    df_encoded = pd.concat([df.drop("track_genre", axis=1), genre_df], axis=1)

    X = df_encoded.drop("popularity", axis=1)
    y = df_encoded["popularity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Music_Popularity_Prediction")

    with mlflow.start_run():
        mlflow.set_tag("developer", "Kordan_Pavlo")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset_version", "1.1")

        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "random_state": 42
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
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        feat_importances.nlargest(15).sort_values().plot(kind='barh', color='skyblue')
        plt.title("Top 15 Feature Importances")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        
        plot_path = "feature_importance.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"✅ Run complete! RMSE: {rmse:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    main()