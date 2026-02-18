import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vector_search import VectorSearch
from src.config import EMBEDDING_MODEL, CLUSTERED_DATA_PATH, EMBEDDINGS_PATH

st.set_page_config(layout="wide")
st.title("🧠 Amazon Review Insight Engine")

clustered_path = ROOT_DIR / CLUSTERED_DATA_PATH
embeddings_path = ROOT_DIR / EMBEDDINGS_PATH

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
