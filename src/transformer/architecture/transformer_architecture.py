from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from ..attention.attention_visualization import AttnViz
from ..pipeline.transformer_outputs import AttentionOutput, TransformerOutput


class TransformerArchitecture:
    def __init__(self, m, f, h=2, l=1):
        self.m = m
        self.f = f
        self.h = h
        self.l = l
        self.e = [TransformerEncoder(m, f, h) for _ in range(l)]
        self.d = [TransformerDecoder(m, f, h) for _ in range(l)]
        self.v = AttnViz()

    def encode(self, s, positions=None):
        x = s
        for e in self.e:
            x = e.forward(x, positions=positions)
        return x

    def decode(self, t, e, positions=None):
        x = t
        for d in self.d:
            x = d.forward(x, e, positions=positions)
        return x

    def forward(self, s, t, source_positions=None, target_positions=None):
        e = self.encode(s, positions=source_positions)
        d = self.decode(t, e, positions=target_positions)
        return TransformerOutput(encoded=e, decoded=d, attention=self.attention())

    def attention(self):
        return AttentionOutput(
            encoder=self.e[-1].attention() if self.e else None,
            decoder_self=self.d[-1].last_self_attention if self.d else None,
            decoder_cross=self.d[-1].last_cross_attention if self.d else None,
        )

    def visualize_attention(self, weights=None, labels=None):
        if weights is None:
            weights = self.attention().get("decoder_self")
        return self.v.forward(weights, labels)

    def summary(self):
        return {
            "architecture": "encoder-decoder transformer",
            "model_size": self.m,
            "feed_forward_size": self.f,
            "heads": self.h,
            "layers": self.l,
            "encoder": "TransformerEncoder",
            "decoder": "TransformerDecoder",
        }


class Arch(TransformerArchitecture):
    pass
