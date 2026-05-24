from dataclasses import dataclass, field
from typing import Any, List


@dataclass(frozen=True)
class AttentionOutput:
    encoder: Any = None
    decoder_self: Any = None
    decoder_cross: Any = None


@dataclass(frozen=True)
class TransformerOutput:
    encoded: Any
    decoded: Any
    logits: Any = None
    attention: AttentionOutput = field(default_factory=AttentionOutput)


@dataclass(frozen=True)
class PipelineSummary:
    architecture: str
    model_size: int
    feed_forward_size: int
    heads: int
    layers: int
    tokenizer: str = "WhitespaceTokenizer"
    embedding: str = "Embedding"


@dataclass(frozen=True)
class PreparedBatch:
    source_ids: List[int]
    target_ids: List[int]
    source_embeddings: List[List[float]]
    source_positions: List[List[float]]
    source_positioned_embeddings: List[List[float]]
    target_embeddings: List[List[float]]
    target_positions: List[List[float]]
    target_positioned_embeddings: List[List[float]]


@dataclass(frozen=True)
class PipelineForwardOutput:
    batch: PreparedBatch
    model_output: TransformerOutput


@dataclass(frozen=True)
class AttentionPreviews:
    encoder: str
    decoder_self: str
    decoder_cross: str


@dataclass(frozen=True)
class PipelineReport:
    summary: PipelineSummary
    forward: PipelineForwardOutput
    attention: AttentionOutput
    attention_previews: AttentionPreviews