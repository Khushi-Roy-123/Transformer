import os
import sys

def main():
    # Prefer src/ directory layout so package imports resolve to src/transformer
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from transformer.pipeline import TransformerPipeline, TransformerPipelineConfig
    from transformer.tokenization import WhitespaceTokenizer

    tokenizer = WhitespaceTokenizer()
    tokenizer.fit([
        'hello world from transformer',
        'world transformer',
    ]).freeze()

    pipeline = TransformerPipeline(
        TransformerPipelineConfig(model_size=32, feed_forward_size=64, heads=4, layers=1),
        tokenizer=tokenizer,
    )

    out = pipeline.forward_text('hello world from transformer', 'world transformer')
    print('encoded len:', len(out.model_output.encoded))
    print('decoded len:', len(out.model_output.decoded))
    print('logits rows:', len(out.model_output.logits or []))

    print('\nGenerated text:')
    print(pipeline.generate('hello world from transformer'))

    print('\nEncoder attention preview:')
    print(pipeline.attention_preview('encoder'))
    print('\nDecoder self-attention preview:')
    print(pipeline.attention_preview('decoder_self'))
    print('\nDecoder cross-attention preview:')
    print(pipeline.attention_preview('decoder_cross'))

if __name__ == '__main__':
    main()
