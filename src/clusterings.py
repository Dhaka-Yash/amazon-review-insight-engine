import umap
import hdbscan
from sklearn.metrics import silhouette_score
from .config import MIN_CLUSTER_SIZE


def reduce_dimensions(embeddings):
    reducer = umap.UMAP(n_neighbors=15, n_components=5)
    return reducer.fit_transform(embeddings)


def cluster_embeddings(reduced_embeddings):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE)
    return clusterer.fit_predict(reduced_embeddings)


def evaluate_clusters(reduced_embeddings, clusters):
    return silhouette_score(reduced_embeddings, clusters)
