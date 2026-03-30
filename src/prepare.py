import argparse
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a Music Popularity data for Prediction model."
    )

    parser.add_argument(
        "--data_input",
        type=str,
        default="data/raw/dataset.csv",
        help="Path to the raw dataset",
    )
    parser.add_argument(
        "--data_output",
        type=str,
        default="data/prepared/",
        help="Path to the save dict",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data_input)

    df.dropna(subset=["artists"], inplace=True)
    df = df[df.popularity != 0].copy()

    drop_cols = ["index", "track_id", "track_name", "album_name", "artists"]
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)

    df["explicit"] = df["explicit"].astype(int)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_genre = encoder.fit_transform(df[["track_genre"]])
    genre_columns = encoder.get_feature_names_out(["track_genre"])
    genre_df = pd.DataFrame(encoded_genre, columns=genre_columns, index=df.index)

    df_encoded = pd.concat([df.drop("track_genre", axis=1), genre_df], axis=1)

    df_train, df_test = train_test_split(df_encoded, test_size=0.2, random_state=42)
    os.makedirs(args.data_output, exist_ok=True)

    df_train.to_csv(args.data_output + "train.csv", index=False)
    df_test.to_csv(args.data_output + "test.csv", index=False)


if __name__ == "__main__":
    main()
