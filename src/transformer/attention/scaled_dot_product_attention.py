E = 2.718281828459045


class SAttn:
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

    def _seq(self, x):
        if not x:
            return [], True
        if isinstance(x[0], list):
            return x, False
        return [x], True

    def forward(self, q, k, v, m=None):
        qs, qs1 = self._seq(q)
        ks, _ = self._seq(k)
        vs, _ = self._seq(v)
        if not qs or not ks or not vs:
            return ([], []) if not qs1 else ([], [])

        os = []
        ws = []
        z = pow(len(qs[0]), 0.5) if qs and qs[0] else 1.0

        for i, qi in enumerate(qs):
            sc = []
            mr = m[i] if m and i < len(m) else None
            for j, kj in enumerate(ks):
                if mr is not None and j < len(mr) and not mr[j]:
                    sc.append(-1e9)
                else:
                    sc.append(self.dot_product(qi, kj) / z)

            w = self.softmax(sc)
            o = [0.0 for _ in range(len(vs[0]))]
            for a, vv in zip(w, vs):
                for j, p in enumerate(vv):
                    o[j] += a * p
            os.append(o)
            ws.append(w)

        if qs1:
            return os[0], ws[0]
        return os, ws

    def d(self, a, b):
        return self.dot_product(a, b)

    def e(self, x):
        return self.exp(x)

    def s(self, v):
        return self.softmax(v)

    def f(self, q, k, v, m=None):
        return self.forward(q, k, v, m)
