E = 2.718281828459045


class BahAttn:
    def __init__(self, h):
        self.h = h
        self.W1 = [[0.1 for _ in range(h)] for _ in range(h)]
        self.W2 = [[0.1 for _ in range(h)] for _ in range(h)]
        self.V = [0.1 for _ in range(h)]

    def dot_product(self, a, b):
        return sum(x * y for x, y in zip(a, b))

    def exp(self, x):
        return pow(E, x)

    def softmax(self, v):
        if not v:
            return []
        m = max(v)
        x = [self.exp(i - m) for i in v]
        t = sum(x)
        if t == 0:
            return [0.0 for _ in x]
        return [i / t for i in x]

    def linear(self, x, m):
        return [sum(x[k] * m[k][j] for k in range(len(x))) for j in range(len(m[0]))]

    def score(self, q, k):
        c = [a + b for a, b in zip(self.linear(q, self.W1), self.linear(k, self.W2))]
        r = [i if i > 0 else 0.0 for i in c]
        return self.dot_product(self.V[:len(c)], r)

    def forward(self, q, v):
        sc = [self.score(q, i) for i in v]
        w = self.softmax(sc)
        o = [0.0 for _ in range(len(v[0]))] if v else []
        for a, vv in zip(w, v):
            for i, p in enumerate(vv):
                o[i] += a * p
        return o, w

    def d(self, a, b):
        return self.dot_product(a, b)

    def e(self, x):
        return self.exp(x)

    def s(self, v):
        return self.softmax(v)

    def l(self, x, m):
        return self.linear(x, m)

    def f(self, q, v):
        return self.forward(q, v)


class BahdanauAttention(BahAttn):
    pass
