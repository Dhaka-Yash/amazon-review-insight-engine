import faiss


class VectorSearch:
    def __init__(self, embeddings):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, model, query, df, k=5):
        query_embedding = model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        return df.iloc[indices[0]][["Score", "full_text"]]
