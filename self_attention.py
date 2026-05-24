from .attention_mechanism import SAttn


class SelfAttention:
    def __init__(self):
        self.w = None
        self.a = SAttn()

    def forward(self, x, m=None):
        o, w = self.a.forward(x, x, x, m)
        self.w = w
        return o

    def f(self, x, m=None):
        return self.forward(x, m)

    def attention_weights(self):
        return self.w
