from .layer_normalisation import LNorm
from .multihead_attention import MHAttn
from .positional_encoding import PEnc


class TransformerDecoder:
    def __init__(self, m, f, h=2):
        self.m = m
        self.f = f
        self.h = h
        self.self_attn = MHAttn(h, m)
        self.cross_attn = MHAttn(h, m)
        self.norm1 = LNorm(m)
        self.norm2 = LNorm(m)
        self.norm3 = LNorm(m)
        self.pos = PEnc()
        self.last_self_attention = None
        self.last_cross_attention = None
        self.W1 = [[((i + 1) * (j + 1)) * 0.01 for j in range(f)] for i in range(m)]
        self.W2 = [[((i + 1) * (j + 1) + 5) * 0.01 for j in range(m)] for i in range(f)]

    def _fix(self, x):
        y = x[:self.m]
        if len(y) < self.m:
            y = y + [0.0 for _ in range(self.m - len(y))]
        return y

    def _add(self, a, b):
        return [x + y for x, y in zip(a, b)]

    def feed_forward(self, x):
        out = []
        for v in x:
            h = []
            for j in range(self.f):
                s = sum(v[i] * self.W1[i][j] for i in range(self.m))
                h.append(s if s > 0 else 0.0)
            y = []
            for i in range(self.m):
                y.append(sum(h[j] * self.W2[j][i] for j in range(self.f)))
            out.append(y)
        return out

    def causal_mask(self, n):
        return [[1 if j <= i else 0 for j in range(n)] for i in range(n)]

    def forward(self, tgt, enc, positions=None):
        tgt = [self._fix(v) for v in tgt]
        enc = [self._fix(v) for v in enc]
        pos = positions if positions is not None else self.pos.forward(len(tgt), self.m)
        x = [self._add(tgt[i], pos[i]) for i in range(len(tgt))]

        cm = self.causal_mask(len(x))
        s, sw = self.self_attn.forward(x, x, x, cm)
        self.last_self_attention = sw
        r1 = [self._add(x[i], s[i]) for i in range(len(x))]
        n1 = [self.norm1.forward(v) for v in r1]

        c, cw = self.cross_attn.forward(n1, enc, enc)
        self.last_cross_attention = cw
        r2 = [self._add(n1[i], c[i]) for i in range(len(n1))]
        n2 = [self.norm2.forward(v) for v in r2]

        y = self.feed_forward(n2)
        r3 = [self._add(n2[i], y[i]) for i in range(len(n2))]
        return [self.norm3.forward(v) for v in r3]


class Dec(TransformerDecoder):
    pass
