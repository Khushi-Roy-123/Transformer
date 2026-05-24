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

## Project Structure

```text
transformer/
│
├── attention_mechanism.py
├── multihead_attention.py
├── self_attention.py
├── bahdanau_attention.py
├── luong_attention.py
├── encoder.py
├── decoder.py
├── positional_encoding.py
├── layer_normalisation.py
├── transformer_architecture.py
├── attention_visualization.py
├── toy_trainer.py
└── __init__.py
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

## Future Improvements

- Tensor abstraction layer
- Manual backpropagation engine
- Tokenizer + inference pipeline
- NumPy acceleration
- Training visualization tools

