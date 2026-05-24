![Transformer architecture diagram](image.png)

# Transformer From Scratch (Pure Python)

A dependencyfree implementation of core Transformer components using only native Python data structures and manual matrix operations.

## Features

- Scaled Dot-Product Attention
- Multi-Head Attention (`Wq`, `Wk`, `Wv`, `Wo`)
- Encoder–Decoder Architecture
- Causal Masking
- Positional Encoding
- Layer Normalization
- Bahdanau & Luong Attention
- Attention Visualization
- Toy Training Pipeline

## Architecture

Input → Positional Encoding → Multi-Head Attention → Add & Norm → Feed Forward → Decoder (Masked Self-Attention + Cross Attention) → Output

## Example

```python
from transformer.transformer_architecture import TransformerArchitecture

arch = TransformerArchitecture(
	model_dim=32,
	feed_forward_dim=64,
	heads=4,
	layers=1
)

output = arch.forward(src, tgt)
print(output)
```

## Limitations

- Pure Python implementation (slow for large inputs)
- No automatic differentiation
- Intended for educational purposes

