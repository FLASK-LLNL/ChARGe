import torch
import numpy as np
from numpy import ndarray
from torch.nn.utils.rnn import pad_sequence
from .tokenizers import SmilesTokenizer


class SmilesEmbedder:
    def __init__(self,
        model_path: str,
        tokenizer: SmilesTokenizer,
        max_len: int | None = None,
        device: torch.device | int | None = None
    ) -> None:
        """
        Args:
            model_path (str): path to trained embedding model
            tokenizer (SmilesTokenizer): a tokenizer class for tokenizing SMILES strings
            max_len (int | None): if specified, limits the number of tokens from the tokenizer to a max length
            device (torch.device | int | None): specifies where the model should live in
        """
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

        self.model = torch.jit.load(model_path).eval()
        if device is not None:
            self.model.cuda(device)

    def pad_input_ids(self, input_ids: list[list[int]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.vocab.get(self.tokenizer.pad_token)
        padded_ids = pad_sequence(
            [torch.tensor(ids) for ids in input_ids],
            batch_first=True,
            padding_value=pad_id,
        )
        if self.max_len is not None:
            padded_ids = padded_ids[:, :self.max_len]
        attn_mask = (padded_ids != pad_id)
        return {'input_ids': padded_ids.to(self.device), 'attention_mask': attn_mask.long().to(self.device)}

    def embed_smiles(self, smiles: list[str]) -> ndarray:
        """
        Args:
            smiles (list[str]): a list of SMILES strings
        Returns:
            Embedding vectors of shape ``[B, d]``.
        """
        assert isinstance(smiles, list), 'Input argument `smiles` must be of type list[str].'
        
        ragged_ids = self.tokenizer(smiles)
        batch = self.pad_input_ids(ragged_ids)
        with torch.inference_mode():
            emb = self.model(batch['input_ids'], batch['attention_mask'])
        return emb.cpu().numpy().astype(np.float32)
