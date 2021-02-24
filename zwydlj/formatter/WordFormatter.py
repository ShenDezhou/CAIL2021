import json
import torch
import numpy as np
import lawa
import lawrouge
import re

class WordFormatter:
    def __init__(self, config, mode):
        self.max_question_len = config.getint("data", "max_question_len")
        self.max_option_len = config.getint("data", "max_option_len")

        self.word2id = json.load(open(config.get("data", "word2id"), "r", encoding="utf8"))
        self.cut = lawa.cut
        self.rouge = lawrouge.Rouge(exclusive=True)

        self.reg = re.compile("[错误|不正确]")

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

    def score(self, option, context, type='p'):
        scores = self.rouge.get_scores([option]*len(context), context, avg=2, ignore_empty=True)
        return scores[type]

    def process(self, data, config, mode, *args, **params):
        context = []
        context_inverse = []
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
                temp_context_inverse = []
                temp_question = []
                q_text = _question['Question']
                temp_question.append(self.convert(q_text+content, self.max_question_len, trucate_head=False))

                if re.search(self.reg, q_text):
                    temp_context_inverse.append(0)
                else:
                    temp_context_inverse.append(1)

                for choice in _question["Choices"]:
                    temp_context.append(self.convert(choice, self.max_option_len))
                for _ in range(4 - len(_question["Choices"])):
                    temp_context.append(self.convert("", self.max_option_len))
                    # temp_context_score.append(0)

                context.append(temp_context)
                question.append(temp_question)
                context_inverse.append(temp_context_inverse)

        question = torch.LongTensor(question)
        context = torch.LongTensor(context)
        context_inverse = torch.FloatTensor(context_inverse)
        if mode != "test":
            label = torch.LongTensor(np.array(label, dtype=np.int32))
            return {"context": context, "if_positive": context_inverse, "question": question, 'label': label, "id": idx}
        else:
            return {"context": context, "if_positive": context_inverse, "question": question, "id": idx}
