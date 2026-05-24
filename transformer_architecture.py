from .decoder import TransformerDecoder
from .encoder import TransformerEncoder
from .attention_visualization import AttnViz


class TransformerArchitecture:
    def __init__(self, m, f, h=2, l=1):
        self.m = m
        self.f = f
        self.h = h
        self.l = l
        self.e = [TransformerEncoder(m, f, h) for _ in range(l)]
        self.d = [TransformerDecoder(m, f, h) for _ in range(l)]
        self.v = AttnViz()

    def encode(self, s):
        x = s
        for e in self.e:
            x = e.forward(x)
        return x

    def decode(self, t, e):
        x = t
        for d in self.d:
            x = d.forward(x, e)
        return x

    def forward(self, s, t):
        e = self.encode(s)
        d = self.decode(t, e)
        return {
            "encoded": e,
            "decoded": d,
        }

    def attention(self):
        return {
            "encoder": self.e[-1].attention() if self.e else None,
            "decoder_self": self.d[-1].last_self_attention if self.d else None,
            "decoder_cross": self.d[-1].last_cross_attention if self.d else None,
        }

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