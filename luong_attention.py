E = 2.718281828459045


class LuoAttn:
    def __init__(self):
        self.w = None

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

    def score(self, q, k):
        return self.dot_product(q, k)

    def forward(self, q, v):
        sc = [self.score(q, i) for i in v]
        w = self.softmax(sc)
        o = [0.0 for _ in range(len(v[0]))] if v else []
        for a, vv in zip(w, v):
            for i, p in enumerate(vv):
                o[i] += a * p
        self.w = w
        return o, w

    def d(self, a, b):
        return self.dot_product(a, b)

    def e(self, x):
        return self.exp(x)

    def s(self, v):
        return self.softmax(v)

    def f(self, q, v):
        return self.forward(q, v)


class LuongAttention(LuoAttn):
    pass
