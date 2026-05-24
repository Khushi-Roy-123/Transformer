from transformer.pipeline import TransformerPipeline, TransformerPipelineConfig

def main():
    pipeline = TransformerPipeline(TransformerPipelineConfig())
    print('Translation demo ready')

if __name__ == '__main__':
    main()
