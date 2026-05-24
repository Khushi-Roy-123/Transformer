from transformer.pipeline import TransformerPipeline


def main():
    tp = TransformerPipeline()
    print("TransformerPipeline created with config:", tp.summary())


if __name__ == "__main__":
    main()
