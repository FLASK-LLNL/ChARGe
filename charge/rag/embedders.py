import torch
import numpy as np
from numpy import ndarray
from .tokenizers import ChemformerTokenizer


class ChemformerEmbedder:
    def __init__(self, model_path: str, vocab_path: str) -> None:
        """
        Args:
            model_path (str): path to saved Chemformer embedding model file
            vocab_path (str): Chemformer-specific vocab file in json format that looks like
                {
                    "properties": {
                        "special_tokens": {...},
                        ...
                    },
                    "vocabulary": [list_of_tokens]
                }
        """
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.model = torch.jit.load(model_path).eval()
        self.tokenizer = ChemformerTokenizer(vocab_path=vocab_path)

    def pad_input_ids(self, input_ids: list[list[int]]) -> tuple[ndarray, ndarray]:
        max_len = max(len(ids) for ids in input_ids)
        pad_id = self.tokenizer.vocab.get(self.tokenizer.pad_token)
        ids = np.full((len(input_ids), max_len), pad_id, dtype=int)
        mask = np.zeros((len(input_ids), max_len), dtype=int)
        for i, row in enumerate(input_ids):
            n = len(row)
            ids[i, :n] = row
            mask[i, :n] = 1
        return ids, mask

    def embed_smiles(self, smiles: list[str]) -> ndarray:
        """
        Args:
            smiles (list[str]): a list of SMILES strings
        Returns:
            Embedding vectors of shape ``[B, d]``.
        """
        assert isinstance(smiles, list)
        
        ragged_ids = self.tokenizer(smiles)
        ids, mask = self.pad_input_ids(ragged_ids)
        with torch.inference_mode():
            emb = self.model(
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long),
            )
        return emb.cpu().numpy().astype(np.float32)
