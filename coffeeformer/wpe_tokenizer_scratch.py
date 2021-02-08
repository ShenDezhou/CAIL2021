from pathlib import Path

from tokenizers import BertWordPieceTokenizer


paths = [str(x) for x in Path("./data/").glob("**/data*eval.csv")]

# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=21_128, min_frequency=0, special_tokens=[
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "<S>",
                "<T>"
                ])


# Save files to disk
tokenizer.save_model("model/wpe", "")