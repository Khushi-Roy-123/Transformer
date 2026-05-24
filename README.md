![Architecture](image.png)

# Transformer From Scratch (Pure Python)

A dependency-free implementation of core Transformer architecture components using only native Python data structures and manual matrix operations.

This project was built to understand how Transformers work internally without relying on frameworks like PyTorch or TensorFlow.

---

## Features

- Scaled Dot-Product Attention
- Multi-Head Attention (`Wq`, `Wk`, `Wv`, `Wo`)
- Encoder–Decoder Transformer Architecture
- Causal Masking for Decoder
- Residual Connections & Layer Normalization
- Positional Encoding
- Bahdanau Attention
- Luong Attention
- Attention Visualization
- Toy Training Pipeline
- High-level orchestration pipeline
- Tokenizer and embedding abstraction
- Typed pipeline outputs

---

## Architecture Flow

```text
Input
 ↓
Positional Encoding
 ↓
Multi-Head Self Attention
 ↓
Add & LayerNorm
 ↓
Feed Forward Network
 ↓
Decoder (Masked Self-Attention + Cross Attention)
 ↓
Output
```

---
## Architecture Overview

This repository provides a from-scratch, dependency-free implementation of Transformer components organized for clarity and learning. The code is located under `src/transformer` and split into focused packages that mirror Transformer concepts.

- **`src/transformer/architecture/`**: high-level encoder/decoder and `transformer_architecture.py` that composes the model (encoder.py, decoder.py, transformer_architecture.py).
- **`src/transformer/attention/`**: core attention mechanisms and utilities (scaled_dot_product_attention.py, multi_head_attention.py, attention_visualization.py).
- **`src/transformer/embeddings/`**: token and positional embedding implementations (token_embedding.py, positional_encoding.py).
- **`src/transformer/tokenization/`**: simple tokenizer abstraction used by examples and pipeline (tokenizer.py).
- **`src/transformer/pipeline/`**: orchestration and typed outputs for running the model end-to-end (transformer_pipeline.py, transformer_outputs.py).
- **`src/transformer/training/`**: lightweight trainer and toy training loops (toy_trainer.py).
- **`src/transformer/utils/`**: small helper utilities for layers and math (layer_utils.py, math_utils.py).
- **`examples/`**: runnable demos (attention_demo.py, translation_demo.py, example_run.py).
- **`tests/`**: unit tests for pipeline and components (tests/test_pipeline.py).

## Project Structure (workspace view)

```text
LICENSE
README.md
examples/
	attention_demo.py
	example_run.py
	translation_demo.py
src/
	transformer/
		__init__.py
		architecture/
			encoder.py
			decoder.py
			transformer_architecture.py
		attention/
			scaled_dot_product_attention.py
			multi_head_attention.py
			attention_visualization.py
		embeddings/
			positional_encoding.py
			token_embedding.py
		pipeline/
			transformer_pipeline.py
			transformer_outputs.py
		tokenization/
			tokenizer.py
		training/
			toy_trainer.py
		utils/
			layer_utils.py
			math_utils.py
tests/
	test_pipeline.py
```

---

## Key Learnings

Building transformers manually helped me understand:

- how scaled attention stabilizes computation,
- why causal masking is necessary,
- how residual connections improve stability,
- how multi-head attention captures different relationships.

---

## Limitations

- Pure Python implementation (slow for large inputs)
- No automatic differentiation
- No GPU acceleration
- Intended for educational and research purposes

---