import re
import json
from abc import ABC, abstractclassmethod


class SmilesTokenizer(ABC):
    """Base class for tokenizing SMILES strings.
    This base class comes with a default tokenization implementaion (based on regular expression)
    but does not have a vocab mapping, i.e., a token-to-id mapping.
    Derived classes must implement token-to-id and id-to-token mappings.
    """
    def __init__(self) -> None:
        self.regex_pattern = \
            r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, smiles: str) -> list[str]:
        return [token for token in self.regex.findall(smiles)]

    @abstractclassmethod
    def _convert_token_to_id(self, token: str) -> int:
        pass

    @abstractclassmethod
    def _convert_id_to_token(self, index: int) -> str:
        pass

    def encode(self, smiles: str) -> list[int]:
        tokens = self.tokenize(smiles)
        return [self._convert_token_to_id(t) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        return ''.join([self._convert_id_to_token(i) for i in ids])

    def batch_encode(self, smiles: list[str]) -> list[list[int]]:
        return [self.encode(smi) for smi in smiles]

    def batch_decode(self, ids_list: list[list[int]]) -> list[str]:
        return [self.decode(ids) for ids in ids_list]
    
    def __call__(self, smiles: str | list[str]) -> list[list[int]]:
        if isinstance(smiles, str):
            return self.batch_encode([smiles])
        elif isinstance(smiles, list):
            return self.batch_encode(smiles)
        else:
            raise ValueError(f'Incompatible input argument: {smiles}')


class ChemformerTokenizer(SmilesTokenizer):
    def __init__(self, vocab_path: str) -> None:
        """
        Args:
            vocab_path (str): Chemformer-specific vocab file in json format that looks like:
                {
                    "properties": {
                        "special_tokens": {...},
                        ...
                    },
                    "vocabulary": [list_of_tokens]
                }
        """
        super().__init__()

        # Load Chemformer-specific vocab file
        with open(vocab_path, 'r') as f:
            vocab_config = json.load(f)
        assert 'vocabulary' in vocab_config and 'properties' in vocab_config, \
            'Vocab file does not have the right Chemformer-specific format.'
        self.vocab = {token: i for i, token in enumerate(vocab_config['vocabulary'])}
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.special_tokens = vocab_config['properties'].get('special_tokens', {})
        self.bos_token = self.special_tokens.get('start', '^')
        self.eos_token = self.special_tokens.get('end', '$')
        self.pad_token = self.special_tokens.get('pad', '<PAD>')
        self.unk_token = self.special_tokens.get('unknown', '?')

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)
    
    def encode(self, smiles: str) -> list[int]:
        # Chemformer sequences are BOS ... EOS, where BOS="^", EOS="&" by default
        tokens = [self.bos_token] + self.tokenize(smiles) + [self.eos_token]
        return [self._convert_token_to_id(t) for t in tokens]