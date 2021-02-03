from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


paths = [str(x) for x in Path("./data/").glob("**/*eval.csv")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=21_128, min_frequency=0, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.post_train()

# Save files to disk
tokenizer.save_model("model/bbpe", "")