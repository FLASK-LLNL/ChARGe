import json
import faiss
import numpy as np
from numpy import ndarray


class ReactionDataRetriever:
    def __init__(self, data_path: str, emb_path: str) -> None:
        self.data_path = data_path
        self.emb_path = emb_path
        
        # Load the data file
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

        # Load the corresponding embedding file
        emb = np.load(emb_path)
        dim = emb.shape[1]
        self.faiss_index = faiss.IndexHNSWFlat(dim, 32)
        self.faiss_index.add(emb)

    def search_similar(self, query: ndarray, k: int) -> list[list[dict]]:
        D, I = self.faiss_index.search(query, k)
        similar = []
        for row in I:
            similar.append([self.data[i] for i in row])
        return similar