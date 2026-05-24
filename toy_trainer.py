class ToyTrainer:
    def __init__(self, arch):
        self.arch = arch

    def train_step(self, src, tgt):
        # placeholder: run forward and return a dummy loss
        out = self.arch.forward(src, tgt)
        # compute a fake loss as sum of all decoded values (just for demonstration)
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
