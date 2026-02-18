from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from .config import EMBEDDING_MODEL


class EmbeddingPipeline:
    def __init__(self):
        print("Loading embedding model...")
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)

    def generate(self, texts):
        print("Generating embeddings...")
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        unique_texts, inverse = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
        print(f"Encoding {len(unique_texts)} unique texts on {self.device}...")
        unique_embeddings = self.model.encode(
            unique_texts.tolist(),
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return unique_embeddings[inverse]

    def save(self, embeddings, path):
        np.save(path, embeddings)
        print("Embeddings saved.")

    def load(self, path):
        return np.load(path)
