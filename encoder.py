from .layer_normalisation import LNorm
from .multihead_attention import MHAttn
from .positional_encoding import PEnc


class TransformerEncoder:
    def __init__(self, m, f, h=2):
        self.m = m
        self.f = f
        self.h = h
        self.attn = MHAttn(h, m)
        self.norm1 = LNorm(m)
        self.norm2 = LNorm(m)
        self.pos = PEnc()
        self.last_attention = None
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

    def forward(self, seq, mask=None):
        seq = [self._fix(v) for v in seq]
        pos = self.pos.forward(len(seq), self.m)
        x = [self._add(seq[i], pos[i]) for i in range(len(seq))]
        a, w = self.attn.forward(x, x, x, mask)
        self.last_attention = w
        r1 = [self._add(x[i], a[i]) for i in range(len(x))]
        n1 = [self.norm1.forward(v) for v in r1]
        y = self.feed_forward(n1)
        r2 = [self._add(n1[i], y[i]) for i in range(len(n1))]
        return [self.norm2.forward(v) for v in r2]

    def attention(self):
        return self.last_attention


class Enc(TransformerEncoder):
    pass
