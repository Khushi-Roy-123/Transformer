from .attention_mechanism import SAttn


class MHAttn:
    def __init__(self, h, m=None):
        self.h = h
        self.m = m
        self.d = None
        self.w = []
        self.a = SAttn()
        self.Wq = []
        self.Wk = []
        self.Wv = []
        self.Wo = None

    def _mat(self, r, c, s=0):
        return [[((i + 1) * (j + 1) + s) * 0.01 for j in range(c)] for i in range(r)]

    def _vec_mat(self, x, m):
        return [sum(x[k] * m[k][j] for k in range(len(x))) for j in range(len(m[0]))]

    def _seq_mat(self, x, m):
        return [self._vec_mat(v, m) for v in x]

    def _fix(self, x, n):
        y = x[:n]
        if len(y) < n:
            y = y + [0.0 for _ in range(n - len(y))]
        return y

    def _seq(self, x):
        if not x:
            return [], True
        if isinstance(x[0], list):
            return x, False
        return [x], True

    def _init(self, n):
        if self.d is None:
            self.d = max(1, (self.m or n) // self.h) if self.h else (self.m or n)
            self.Wq = [self._mat(n, self.d, i) for i in range(self.h)]
            self.Wk = [self._mat(n, self.d, i + 7) for i in range(self.h)]
            self.Wv = [self._mat(n, self.d, i + 13) for i in range(self.h)]
            self.Wo = self._mat(self.h * self.d, self.m or n, 19)

    def forward(self, q, k, v, m=None):
        qs, q1 = self._seq(q)
        ks, _ = self._seq(k)
        vs, _ = self._seq(v)
        if not qs or not ks or not vs:
            return ([], []) if q1 else ([], [])

        n = len(self._fix(qs[0], len(qs[0])))
        self._init(n)
        hs = []
        self.w = []
        qx = [self._fix(x, n) for x in qs]
        kx = [self._fix(x, n) for x in ks]
        vx = [self._fix(x, n) for x in vs]

        for i in range(self.h):
            qh = self._seq_mat(qx, self.Wq[i])
            kh = self._seq_mat(kx, self.Wk[i])
            vh = self._seq_mat(vx, self.Wv[i])
            out, wt = self.a.forward(qh, kh, vh, m)
            hs.append(out)
            self.w.append(wt)

        merged = []
        for i in range(len(hs[0])):
            row = []
            for h in range(self.h):
                row.extend(hs[h][i])
            merged.append(self._vec_mat(row, self.Wo))

        if q1:
            return merged[0], self.w
        return merged, self.w

    def f(self, q, k, v, m=None):
        return self.forward(q, k, v, m)

    def dot_product(self, q, k, v, m=None):
        return self.forward(q, k, v, m)


class MultiHeadAttention(MHAttn):
    pass
