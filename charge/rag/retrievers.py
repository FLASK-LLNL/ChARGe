import json
import faiss
import numpy as np
from numpy import ndarray
from typing import Any


class FaissDataRetriever:
    def __init__(self, data_path: str, emb_path: str, data_format: str = 'json') -> None:
        """
        Args:
            data_path (str): path to data file for retrieval. Must be iterable (e.g., a list)
            emb_path (str): path to npy file containing the embedding vectors for 'data_path'
            data_format (str): data file format for 'data_path' (default: 'json')
        """
        self.data_path = data_path
        self.emb_path = emb_path
        self.data_format = data_format
        
        # Load the data file into an iterable
        match data_format:
            case 'json':
                self.data = self._load_json(data_path)
            case _:
                raise NotImplementedError

        # Load the embedding file and set up the FAISS index
        emb = np.load(emb_path)
        dim = emb.shape[1]
        self.faiss_index = faiss.IndexHNSWFlat(dim, 32)
        self.faiss_index.add(emb)

    def _load_json(self, filename: str) -> list[dict]:
        with open(filename, 'r') as f:
            data = [json.loads(line) for line in f]
        return data

    def search_similar(self, query: ndarray, k: int) -> tuple[list[list[float]], list[list[int]], list[list[Any]]]:
        D, I = self.faiss_index.search(query, k)
        similar = []
        for row in I:
            similar.append([self.data[i] for i in row])
        return D.tolist(), I.tolist(), similar
