from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

class ChineseDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = ByteLevelBPETokenizer(
            "./model/bbpe/vocab.json",
            "./model/bbpe/merges.txt",
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)
        # or use the RobertaTokenizer from `transformers` directly.

        self.examples = []

        src_files = Path("./data/").glob("*_eval.csv") if evaluate else Path("./data/").glob("*_eval.csv")
        for src_file in src_files:
            print("🔥", src_file)
            with open(src_file,'r',encoding='utf-8') as f:
                for index, line in enumerate(f):
                    self.examples += [x.ids for x in tokenizer.encode_batch(line)]
                    if index % 10000==0:
                        print(src_file,index//10000)



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])

