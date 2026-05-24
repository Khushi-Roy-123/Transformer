class ToyTrainer:
    def __init__(self, arch):
        self.arch = arch

    def train_step(self, src, tgt):
        out = self.arch.forward(src, tgt)
        dec = out.decoded
        s = 0.0
        for v in dec:
            if isinstance(v, list):
                s += sum(v)
        return s

    def fit(self, dataset, steps=1):
        for _ in range(steps):
            for src, tgt in dataset:
                self.train_step(src, tgt)
