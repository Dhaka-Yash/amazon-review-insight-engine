from src.config import *
from src.data_processing import load_and_clean_data
from src.embeddings import EmbeddingPipeline
from src.clusterings import (
    reduce_dimensions,
    cluster_embeddings,
    evaluate_clusters,
)
from src.trend_analysis import compute_trends
from src.insight_engine import cluster_impact
import os


def run():

    # Step 1: Load + Clean
    df = load_and_clean_data(DATA_PATH)
    df.to_csv(CLEAN_DATA_PATH, index=False)

    # Step 2: Embeddings
    embedder = EmbeddingPipeline()
    if os.path.exists(EMBEDDINGS_PATH):
        cached_embeddings = embedder.load(EMBEDDINGS_PATH)
        if cached_embeddings.shape[0] == len(df):
            print("Using cached embeddings.")
            embeddings = cached_embeddings
        else:
            print("Cached embeddings shape mismatch. Regenerating...")
            embeddings = embedder.generate(df["full_text"].tolist())
            embedder.save(embeddings, EMBEDDINGS_PATH)
    else:
        embeddings = embedder.generate(df["full_text"].tolist())
        embedder.save(embeddings, EMBEDDINGS_PATH)

    # Step 3: Clustering
    reduced = reduce_dimensions(embeddings)
    clusters = cluster_embeddings(reduced)
    df["cluster"] = clusters

    score = evaluate_clusters(reduced, clusters)
    print("Silhouette Score:", score)

    # Step 4: Save clustered data
    df.to_csv(CLUSTERED_DATA_PATH, index=False)

    # Step 5: Business insights
    print("\nCluster Impact (Mean Rating):")
    print(cluster_impact(df))

    # Step 6: Trends
    trend = compute_trends(df)
    print("\nTrend Snapshot:")
    print(trend.tail())

    print("\nPipeline Completed Successfully.")


if __name__ == "__main__":
    run()
