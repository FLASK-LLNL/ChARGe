import pickle
import json
import faiss
from numpy import ndarray


class ReactionDataRetriever:
    def __init__(self, data_path: str, index_path: str) -> None:
        self.data_path = data_path
        self.index_path = index_path
        
        # Load the data file corresponding to the FAISS index
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

        # Load the trained FAISS index object
        with open(index_path, 'rb') as f:
            self.faiss_index = pickle.load(f)

    def search_similar(self, query: ndarray, k: int) -> list[list[dict]]:
        D, I = self.faiss_index.search(query, k)
        similar = []
        for row in I:
            similar.append([self.data[i] for i in row])
        return similar