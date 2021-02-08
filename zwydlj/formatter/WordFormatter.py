import json
import torch
import numpy as np
import lawa


class WordFormatter:
    def __init__(self, config, mode):
        self.max_question_len = config.getint("data", "max_question_len")
        self.max_option_len = config.getint("data", "max_option_len")

        self.word2id = json.load(open(config.get("data", "word2id"), "r", encoding="utf8"))
        self.cut = lawa.cut

    def convert_tokens_to_ids(self, tokens):
        arr = []
        for a in range(0, len(tokens)):
            if tokens[a] in self.word2id:
                arr.append(self.word2id[tokens[a]])
            else:
                arr.append(self.word2id["UNK"])
        return arr

    def convert(self, tokens, max_seq_len, trucate_head=False):

        tokens = list(self.cut(tokens))
        if trucate_head:
            tokens = tokens[len(tokens) - max_seq_len:]
        else:
            tokens = tokens[:max_seq_len]
        ids = self.convert_tokens_to_ids(tokens)
        if len(ids) < max_seq_len:
            ids = ids + ([0] * (max_seq_len - len(ids)))
        return ids

    def process(self, data, config, mode, *args, **params):
        context = []
        question = []
        label = []
        idx = []

        for temp_data in data:
            content = temp_data["Content"]
            questions = temp_data["Questions"]
            for _question in questions:
                idx.append(_question["Q_id"])

                if mode != "test":
                    label_x = "ABCD".find(_question["Answer"])
                    label.append(label_x)

                temp_context = []
                temp_question = []

                temp_question.append(self.convert(content, self.max_question_len, trucate_head=True))
                for choice in _question["Choices"]:
                    temp_context.append(self.convert(choice, self.max_option_len))
                for _ in range(4 - len(_question["Choices"])):
                    temp_context.append(self.convert("", self.max_option_len))

                context.append(temp_context)
                question.append(temp_question)

        question = torch.LongTensor(question)
        context = torch.LongTensor(context)
        if mode != "test":
            label = torch.LongTensor(np.array(label, dtype=np.int32))
            return {"context": context, "question": question, 'label': label, "id": idx}
        else:
            return {"context": context, "question": question, "id": idx}
