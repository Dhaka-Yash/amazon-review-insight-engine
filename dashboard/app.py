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

st.set_page_config(layout="wide")
st.title("🧠 Amazon Review Insight Engine")

default_artifacts_dir = ROOT_DIR / "data"
artifacts_dir = Path(os.getenv("REVIEW_ARTIFACTS_DIR", str(default_artifacts_dir))).expanduser()
clustered_path = artifacts_dir / "clustered_reviews.csv"
embeddings_path = artifacts_dir / "embeddings.npy"

if not clustered_path.exists() or not embeddings_path.exists():
    st.error("Missing pipeline artifacts. Generate files first, then rerun Streamlit.")
    st.code(
        "\n".join(
            [
                f"Expected: {clustered_path}",
                f"Expected: {embeddings_path}",
                "Run pipeline: python main.py --artifacts-dir <folder>",
                "Set env var: $env:REVIEW_ARTIFACTS_DIR='<folder>'",
            ]
        )
    )
    st.stop()

df = pd.read_csv(clustered_path)
embeddings = np.load(embeddings_path)

model = SentenceTransformer(EMBEDDING_MODEL)
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
