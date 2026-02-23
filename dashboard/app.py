import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vector_search import VectorSearch
from src.config import EMBEDDING_MODEL
from src.data_processing import load_and_clean_data
from src.embeddings import EmbeddingPipeline
from src.clusterings import reduce_dimensions, cluster_embeddings

st.set_page_config(layout="wide")
st.title("🧠 Amazon Review Insight Engine")

default_artifacts_dir = ROOT_DIR / "data"
artifacts_dir = Path(os.getenv("REVIEW_ARTIFACTS_DIR", str(default_artifacts_dir))).expanduser()
artifacts_dir.mkdir(parents=True, exist_ok=True)

data_path = Path(os.getenv("REVIEW_DATA_PATH", str(ROOT_DIR / "data" / "Reviews.csv"))).expanduser()
clustered_path = artifacts_dir / "clustered_reviews.csv"
embeddings_path = artifacts_dir / "embeddings.npy"
clean_path = artifacts_dir / "clean_reviews.csv"


@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_data(show_spinner=False)
def ensure_artifacts(clustered_file: str, embeddings_file: str, clean_file: str, source_data: str):
    clustered = Path(clustered_file)
    embeddings = Path(embeddings_file)
    clean = Path(clean_file)
    source = Path(source_data)

    if clustered.exists() and embeddings.exists():
        return

    if not source.exists():
        raise FileNotFoundError(
            f"Source data not found at {source}. Set REVIEW_DATA_PATH to Reviews.csv."
        )

    df_local = load_and_clean_data(str(source))
    df_local.to_csv(clean, index=False)

    embedder = EmbeddingPipeline()
    generated_embeddings = embedder.generate(df_local["full_text"].tolist())
    embedder.save(generated_embeddings, embeddings)

    reduced = reduce_dimensions(generated_embeddings)
    clusters = cluster_embeddings(reduced)
    df_local["cluster"] = clusters
    df_local.to_csv(clustered, index=False)


try:
    with st.spinner("Preparing data artifacts..."):
        ensure_artifacts(
            str(clustered_path),
            str(embeddings_path),
            str(clean_path),
            str(data_path),
        )
except Exception as exc:
    st.error("Could not prepare data artifacts for the app.")
    st.code(
        "\n".join(
            [
                f"Artifacts dir: {artifacts_dir}",
                f"Data path: {data_path}",
                f"Error: {exc}",
            ]
        )
    )
    st.stop()

df = pd.read_csv(clustered_path)
embeddings = np.load(embeddings_path)
model = load_model()
search_engine = VectorSearch(embeddings)

# Semantic Search
st.header("🔍 Semantic Search")
query = st.text_input("Search reviews")

if query:
    results = search_engine.search(model, query, df)
    st.write(results)

# Cluster Overview
st.header("📊 Top Recurring Issues")
st.bar_chart(df["cluster"].value_counts().head(10))

# Cluster Impact
st.header("⭐ Cluster Average Ratings")
st.bar_chart(df.groupby("cluster")["Score"].mean())

# Trend Analysis
st.header("📈 Trends Over Time")
df["YearMonth"] = df["YearMonth"].astype(str)
trend = df.groupby(["YearMonth", "cluster"]).size().unstack(fill_value=0)
st.line_chart(trend)
