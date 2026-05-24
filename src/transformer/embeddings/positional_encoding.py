PI = 3.141592653589793
TWO_PI = 2.0 * PI


class PEnc:
    def wrap_angle(self, x):
        while x > PI:
            x -= TWO_PI
        while x < -PI:
            x += TWO_PI
        return x

    def sin(self, x, t=10):
        x = self.wrap_angle(x)
        r = 0.0
        p = x
        g = 1.0
        f = 1.0
        for n in range(t):
            if n == 0:
                r += p
            else:
                f *= (2 * n) * (2 * n + 1)
                p *= x * x
                g *= -1.0
                r += g * p / f
        return r

    def cos(self, x, t=10):
        x = self.wrap_angle(x)
        r = 1.0
        p = 1.0
        g = -1.0
        f = 1.0
        for n in range(1, t):
            f *= (2 * n - 1) * (2 * n)
            p *= x * x
            r += g * p / f
            g *= -1.0
        return r

    def forward(self, l, d):
        out = []
        for p in range(l):
            r = []
            for i in range(d):
                a = pow(10000.0, -(i // 2) * 2.0 / d) if d else 1.0
                x = p * a
                if i % 2 == 0:
                    r.append(self.sin(x))
                else:
                    r.append(self.cos(x))
            out.append(r)
        return out

    def w(self, x):
        return self.wrap_angle(x)

    def s(self, x, t=10):
        return self.sin(x, t)

    def c(self, x, t=10):
        return self.cos(x, t)


class PositionalEncoding(PEnc):
    pass
