import os
from argparse import ArgumentParser
from pathlib import Path

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


def resolve_artifact_paths(artifacts_dir: str | None):
    if artifacts_dir:
        output_dir = Path(artifacts_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        clean_path = output_dir / "clean_reviews.csv"
        embeddings_path = output_dir / "embeddings.npy"
        clustered_path = output_dir / "clustered_reviews.csv"
        return clean_path, embeddings_path, clustered_path

    return Path(CLEAN_DATA_PATH), Path(EMBEDDINGS_PATH), Path(CLUSTERED_DATA_PATH)


def run(data_path: str = DATA_PATH, artifacts_dir: str | None = None):
    clean_path, embeddings_path, clustered_path = resolve_artifact_paths(artifacts_dir)

    # Step 1: Load + Clean
    df = load_and_clean_data(data_path)
    df.to_csv(clean_path, index=False)

    # Step 2: Embeddings
    embedder = EmbeddingPipeline()
    if os.path.exists(embeddings_path):
        cached_embeddings = embedder.load(embeddings_path)
        if cached_embeddings.shape[0] == len(df):
            print("Using cached embeddings.")
            embeddings = cached_embeddings
        else:
            print("Cached embeddings shape mismatch. Regenerating...")
            embeddings = embedder.generate(df["full_text"].tolist())
            embedder.save(embeddings, embeddings_path)
    else:
        embeddings = embedder.generate(df["full_text"].tolist())
        embedder.save(embeddings, embeddings_path)

    # Step 3: Clustering
    reduced = reduce_dimensions(embeddings)
    clusters = cluster_embeddings(reduced)
    df["cluster"] = clusters

    score = evaluate_clusters(reduced, clusters)
    print("Silhouette Score:", score)

    # Step 4: Save clustered data
    df.to_csv(clustered_path, index=False)

    # Step 5: Business insights
    print("\nCluster Impact (Mean Rating):")
    print(cluster_impact(df))

    # Step 6: Trends
    trend = compute_trends(df)
    print("\nTrend Snapshot:")
    print(trend.tail())

    print("\nSaved files:")
    print(f"- {clean_path.resolve()}")
    print(f"- {embeddings_path.resolve()}")
    print(f"- {clustered_path.resolve()}")
    print("\nTo run Streamlit with external artifacts, set REVIEW_ARTIFACTS_DIR to this folder.")

    print("\nPipeline Completed Successfully.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-path", default=DATA_PATH, help="Path to input Reviews.csv")
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help=(
            "Directory where clean_reviews.csv, embeddings.npy, and "
            "clustered_reviews.csv are saved."
        ),
    )
    args = parser.parse_args()
    run(data_path=args.data_path, artifacts_dir=args.artifacts_dir)
