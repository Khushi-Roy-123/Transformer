from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from ..attention.attention_visualization import AttnViz
from ..embeddings.token_embedding import Embedding
from ..embeddings.positional_encoding import PositionalEncoding
from ..tokenization.tokenizer import WhitespaceTokenizer
from ..training.toy_trainer import ToyTrainer
from ..architecture.transformer_architecture import TransformerArchitecture
from .transformer_outputs import (
    AttentionOutput,
    AttentionPreviews,
    PipelineForwardOutput,
    PipelineReport,
    PipelineSummary,
    PreparedBatch,
    TransformerOutput,
)


def average_attention_heads(weights: Sequence[Sequence[Sequence[float]]]) -> List[List[float]]:
    if not weights:
        return []
    head_count = len(weights)
    first_head = weights[0]
    if not first_head or not first_head[0]:
        return []
    row_count = len(first_head)
    column_count = len(first_head[0])
    for head in weights:
        if len(head) != row_count:
            raise ValueError("attention heads must have matching row counts")
        for row in head:
            if len(row) != column_count:
                raise ValueError("attention heads must have matching column counts")

    out: List[List[float]] = []
    for row_index in range(row_count):
        row: List[float] = []
        for column_index in range(column_count):
            total = 0.0
            for head_index in range(head_count):
                total += weights[head_index][row_index][column_index]
            row.append(total / head_count)
        out.append(row)
    return out


def attention_keys() -> Tuple[str, str, str]:
    return "encoder", "decoder_self", "decoder_cross"


@dataclass(frozen=True)
class TransformerPipelineConfig:
    model_size: int = 32
    feed_forward_size: int = 64
    heads: int = 4
    layers: int = 1
    attention_width: int = 24
    max_generation_steps: int = 16
    freeze_vocab: bool = True


class TransformerPipeline:
    def __init__(
        self,
        config: Optional[TransformerPipelineConfig] = None,
        tokenizer: Optional[WhitespaceTokenizer] = None,
        embedding: Optional[Embedding] = None,
        architecture: Optional[TransformerArchitecture] = None,
        trainer: Optional[ToyTrainer] = None,
        visualizer: Optional[AttnViz] = None,
        positional_encoding: Optional[PositionalEncoding] = None,
    ):
        self.config = config or TransformerPipelineConfig()
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        if self.config.freeze_vocab:
            self.tokenizer.freeze()
        self.embedding = embedding or Embedding(self.tokenizer.vocab_size, self.config.model_size)
        self.positional_encoding = positional_encoding or PositionalEncoding()
        self.architecture = architecture or TransformerArchitecture(
            self.config.model_size,
            self.config.feed_forward_size,
            self.config.heads,
            self.config.layers,
        )
        self.trainer = trainer or ToyTrainer(self.architecture)
        self.visualizer = visualizer or AttnViz()

    def tokenize(self, text: str) -> List[int]:
        ids = self.tokenizer.encode(text)
        return ids

    def detokenize(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids)

    def embed(self, ids: Sequence[int]) -> List[List[float]]:
        return self.embedding.forward(ids)

    def position(self, length: int) -> List[List[float]]:
        return self.positional_encoding.forward(length, self.config.model_size)

    def apply_position(self, embeddings: Sequence[Sequence[float]]) -> List[List[float]]:
        positions = self.position(len(embeddings))
        positioned = []
        for index, row in enumerate(embeddings):
            positioned.append([value + positions[index][column] for column, value in enumerate(row)])
        return positioned

    def prepare(self, src_ids: Sequence[int], tgt_ids: Sequence[int]) -> PreparedBatch:
        src = self.embed(src_ids)
        tgt = self.embed(tgt_ids)
        src_pos = self.position(len(src))
        tgt_pos = self.position(len(tgt))
        src_positioned = self.apply_position(src)
        tgt_positioned = self.apply_position(tgt)
        return PreparedBatch(list(src_ids), list(tgt_ids), src, src_pos, src_positioned, tgt, tgt_pos, tgt_positioned)

    def encode(self, src_ids: Sequence[int]) -> List[List[float]]:
        source_embeddings = self.embed(src_ids)
        source_positions = self.position(len(source_embeddings))
        return self.architecture.encode(source_embeddings, positions=source_positions)

    def decode(self, tgt_ids: Sequence[int], encoded: Sequence[Sequence[float]]) -> List[List[float]]:
        target_embeddings = self.embed(tgt_ids)
        target_positions = self.position(len(target_embeddings))
        return self.architecture.decode(target_embeddings, encoded, positions=target_positions)

    def forward(self, src_ids: Sequence[int], tgt_ids: Sequence[int]) -> PipelineForwardOutput:
        payload = self.prepare(src_ids, tgt_ids)
        model_output = self.architecture.forward(
            payload.source_embeddings,
            payload.target_embeddings,
            source_positions=payload.source_positions,
            target_positions=payload.target_positions,
        )
        if not isinstance(model_output, TransformerOutput):
            raise TypeError("TransformerArchitecture.forward must return TransformerOutput")
        logits = [self.embedding.project(row) for row in model_output.decoded]
        model_output = TransformerOutput(
            encoded=model_output.encoded,
            decoded=model_output.decoded,
            logits=logits,
            attention=model_output.attention,
        )
        return PipelineForwardOutput(batch=payload, model_output=model_output)

    def run(self, src_ids: Sequence[int], tgt_ids: Sequence[int]) -> PipelineForwardOutput:
        return self.forward(src_ids, tgt_ids)

    def train(self, dataset: Iterable[Tuple[Sequence[int], Sequence[int]]], steps: int = 1):
        embedded_dataset = [(self.embed(src), self.embed(tgt)) for src, tgt in dataset]
        self.trainer.fit(embedded_dataset, steps=steps)
        return self

    def attention(self) -> AttentionOutput:
        return self.architecture.attention()

    def attention_preview(self, kind: str = "decoder_self", labels: Optional[Sequence[str]] = None, width: Optional[int] = None) -> str:
        weights = getattr(self.attention(), kind)
        return self.visualizer.render(
            average_attention_heads(weights),
            labels=labels,
            width=width or self.config.attention_width,
        )

    def attention_report(self, labels: Optional[Sequence[str]] = None, width: Optional[int] = None) -> AttentionPreviews:
        return AttentionPreviews(
            encoder=self.attention_preview("encoder", labels=labels, width=width),
            decoder_self=self.attention_preview("decoder_self", labels=labels, width=width),
            decoder_cross=self.attention_preview("decoder_cross", labels=labels, width=width),
        )

    def summary(self) -> PipelineSummary:
        return PipelineSummary(
            architecture=self.architecture.summary()["architecture"],
            model_size=self.config.model_size,
            feed_forward_size=self.config.feed_forward_size,
            heads=self.config.heads,
            layers=self.config.layers,
        )

    def build_report(self, src_ids: Sequence[int], tgt_ids: Sequence[int], labels: Optional[Sequence[str]] = None) -> PipelineReport:
        result = self.forward(src_ids, tgt_ids)
        return PipelineReport(
            summary=self.summary(),
            forward=result,
            attention=self.attention(),
            attention_previews=self.attention_report(labels=labels),
        )

    def forward_text(self, source_text: str, target_text: str) -> PipelineForwardOutput:
        return self.forward(self.tokenize(source_text), self.tokenize(target_text))

    def generate(self, source_text: str, max_steps: Optional[int] = None) -> str:
        source_ids = self.tokenize(source_text)
        generated_ids = [self.tokenizer.bos_id]
        limit = max_steps if max_steps is not None else self.config.max_generation_steps

        for _ in range(limit):
            output = self.forward(source_ids, generated_ids)
            # simple greedy: pick argmax from logits of last position
            last_logits = output.model_output.logits[-1]
            next_id = self.embedding.argmax(last_logits)
            if next_id == self.tokenizer.eos_id:
                break
            generated_ids.append(next_id)

        return self.detokenize(generated_ids)
