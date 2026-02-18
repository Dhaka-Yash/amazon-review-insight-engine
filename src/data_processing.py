import pandas as pd
from .config import SAMPLE_SIZE


def load_and_clean_data(path):
    print("Loading dataset...")

    df = pd.read_csv(path)

    sample_size = min(SAMPLE_SIZE, len(df))
    df = df.sample(sample_size, random_state=42)
    df = df.dropna(subset=["Text"])
    df = df[df["Text"].str.len() > 40]

    time_as_epoch = pd.to_numeric(df["Time"], errors="coerce")
    parsed_epoch = pd.to_datetime(time_as_epoch, unit="s", errors="coerce")
    parsed_string = pd.to_datetime(df["Time"], errors="coerce")
    df["Time"] = parsed_epoch.fillna(parsed_string)
    df["YearMonth"] = df["Time"].dt.to_period("M")

    df["full_text"] = df["Summary"].fillna("") + ". " + df["Text"]

    print("Data cleaned successfully.")
    return df
