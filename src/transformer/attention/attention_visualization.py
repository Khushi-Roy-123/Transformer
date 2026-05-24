class AttnViz:
    def _seq(self, w):
        if not w:
            return []
        if isinstance(w[0], list):
            return w
        return [w]

    def _bar(self, x, width):
        n = int(round(max(0.0, min(1.0, x)) * width))
        return "#" * n + "." * (width - n)

    def forward(self, weights, labels=None, width=24):
        rows = self._seq(weights)
        if not rows:
            return ""

        if labels is None:
            labels = [str(i) for i in range(len(rows))]

        out = []
        head = "     " + " ".join(str(i).rjust(width) for i in range(len(rows[0])))
        out.append(head)
        for i, row in enumerate(rows):
            name = labels[i] if i < len(labels) else str(i)
            parts = [self._bar(v, width) for v in row]
            out.append(name.ljust(4) + " " + " ".join(parts))
        return "\n".join(out)

    def render(self, weights, labels=None, width=24):
        return self.forward(weights, labels, width)

    def heatmap(self, weights, labels=None, width=24):
        return self.forward(weights, labels, width)
