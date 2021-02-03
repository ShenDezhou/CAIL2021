from pathlib import Path

from sentencepiece import SentencePieceTrainer


paths = [str(x) for x in Path("./data/").glob("**/*train.csv")]


# Customize training
SentencePieceTrainer.train(input=paths, model_prefix='model/spbpe/spiece',  vocab_size=21_128, user_defined_symbols=[])

