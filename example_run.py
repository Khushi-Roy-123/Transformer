import sys

def embed(ids, m):
    out = []
    for token in ids:
        row = []
        for i in range(m):
            row.append((token * (i + 1)) / 100.0)
        out.append(row)
    return out

def main():
    sys.path.insert(0, r".")
    from transformer.transformer_architecture import TransformerArchitecture
    from transformer.attention_visualization import AttnViz

    m = 32
    arch = TransformerArchitecture(m, 64, 4, 1)

    src = embed([1, 2, 3, 4], m)
    tgt = embed([2, 3], m)

    out = arch.forward(src, tgt)
    print('encoded len:', len(out['encoded']))
    print('decoded len:', len(out['decoded']))

    viz = AttnViz()
    attn = arch.attention()

    def avg_heads(weights):
        if not weights:
            return []
        if not isinstance(weights[0], list) or not weights[0]:
            return []
        if not isinstance(weights[0][0], list):
            return weights
        heads = len(weights)
        rows = len(weights[0])
        cols = len(weights[0][0])
        out = []
        for i in range(rows):
            row = []
            for j in range(cols):
                s = 0.0
                for h in range(heads):
                    s += weights[h][i][j]
                row.append(s / heads)
            out.append(row)
        return out

    print('\nEncoder attention preview:')
    print(viz.render(avg_heads(attn.get('encoder'))))
    print('\nDecoder self-attention preview:')
    print(viz.render(avg_heads(attn.get('decoder_self'))))
    print('\nDecoder cross-attention preview:')
    print(viz.render(avg_heads(attn.get('decoder_cross'))))

if __name__ == '__main__':
    main()
