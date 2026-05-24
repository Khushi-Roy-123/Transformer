class LNorm:
    def __init__(self, n, e=1e-5):
        self.n = n
        self.e = e
        self.g = [1.0 for _ in range(n)]
        self.b = [0.0 for _ in range(n)]

    def mean(self, x):
        return sum(x) / len(x) if x else 0.0

    def variance(self, x, m):
        if not x:
            return 0.0
        return sum((i - m) ** 2 for i in x) / len(x)

    def forward(self, x):
        m = self.mean(x)
        v = self.variance(x, m)
        d = (v + self.e) ** 0.5
        z = [(i - m) / d for i in x]
        return [self.g[i] * z[i] + self.b[i] for i in range(len(x))]

    def m(self, x):
        return self.mean(x)

    def v(self, x, m):
        return self.variance(x, m)

    def f(self, x):
        return self.forward(x)


class LayerNormalization(LNorm):
    pass
