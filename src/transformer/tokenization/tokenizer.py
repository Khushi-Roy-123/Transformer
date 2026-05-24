from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass
class WhitespaceTokenizer:
    lowercase: bool = True
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    vocab: Dict[str, int] = field(default_factory=dict)
    inverse_vocab: Dict[int, str] = field(default_factory=dict)
    frozen: bool = False

    def __post_init__(self) -> None:
        if not self.vocab:
            self.vocab = {
                self.pad_token: 0,
                self.bos_token: 1,
                self.eos_token: 2,
                self.unk_token: 3,
            }
        self.inverse_vocab = {index: token for token, index in self.vocab.items()}

    @property
    def pad_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.vocab[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _normalize(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def _tokenize(self, text: str) -> List[str]:
        normalized = self._normalize(text).strip()
        if not normalized:
            return []
        return normalized.split()

    def _ensure_token(self, token: str) -> int:
        if token not in self.vocab:
            if self.frozen:
                return self.unk_id
            token_id = len(self.vocab)
            self.vocab[token] = token_id
            self.inverse_vocab[token_id] = token
        return self.vocab[token]

    def fit(self, texts: Sequence[str]) -> "WhitespaceTokenizer":
        for text in texts:
            for token in self._tokenize(text):
                self._ensure_token(token)
        return self

    def freeze(self) -> "WhitespaceTokenizer":
        self.frozen = True
        return self

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self._tokenize(text)
        ids = [self._ensure_token(token) for token in tokens]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in ids:
            token = self.inverse_vocab.get(token_id, self.unk_token)
            if skip_special_tokens and token in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def batch_encode(self, texts: Sequence[str], add_special_tokens: bool = True) -> List[List[int]]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def batch_decode(self, batch_ids: Sequence[Sequence[int]], skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]
