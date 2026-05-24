from typing import List, Optional, Sequence, Set


class Embedding:
    def __init__(self, vocab_size: int, model_size: int, weights: Optional[List[List[float]]] = None):
        self.vocab_size = max(1, vocab_size)
        self.model_size = max(1, model_size)
        self.weights = weights or self._init_weights(self.vocab_size, self.model_size)

    def _init_weights(self, vocab_size: int, model_size: int) -> List[List[float]]:
        weights = []
        for token_id in range(vocab_size):
            row = []
            for i in range(model_size):
                base = ((token_id + 1) * (i + 3)) % 97
                row.append((base / 97.0) - 0.5)
            weights.append(row)
        return weights

    def ensure_size(self, vocab_size: int) -> None:
        if vocab_size <= self.vocab_size:
            return
        for token_id in range(self.vocab_size, vocab_size):
            row = []
            for i in range(self.model_size):
                base = ((token_id + 1) * (i + 3)) % 97
                row.append((base / 97.0) - 0.5)
            self.weights.append(row)
        self.vocab_size = vocab_size

    def forward(self, ids: Sequence[int]) -> List[List[float]]:
        if not ids:
            return []
        upper_bound = max(ids)
        if upper_bound >= self.vocab_size:
            raise ValueError("token id exceeds embedding vocabulary size")
        return [self.weights[token_id][:] for token_id in ids]

    def argmax(self, scores: Sequence[float], exclude_ids: Optional[Set[int]] = None) -> int:
        best_index = 0
        best_score = float("-inf")
        excluded = exclude_ids or set()
        for index, score in enumerate(scores):
            if index in excluded:
                continue
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def project(self, vector: Sequence[float]) -> List[float]:
        scores = []
        for row in self.weights:
            score = 0.0
            for value, weight in zip(vector, row):
                score += value * weight
            scores.append(score)
        return scores

    def nearest(self, vector: Sequence[float], exclude_ids: Optional[Set[int]] = None) -> int:
        return self.argmax(self.project(vector), exclude_ids=exclude_ids)