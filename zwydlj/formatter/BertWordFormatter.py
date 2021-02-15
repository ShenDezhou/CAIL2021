import json
import torch
import numpy as np
import os
#from pytorch_pretrained_bert import BertTokenizer
from transformers import BertTokenizer

class BertWordFormatter:
    def __init__(self, config, mode):
        self.max_question_len = config.getint("data", "max_question_len")
        self.max_option_len = config.getint("data", "max_option_len")

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))

    def convert_tokens_to_ids(self, tokens):
        arr = []
        for a in range(0, len(tokens)):
            if tokens[a] in self.word2id:
                arr.append(self.word2id[tokens[a]])
            else:
                arr.append(self.word2id["UNK"])
        return arr

    def convert(self, tokens, max_seq_len, trucate_head=False):
        tokens = "".join(tokens)
        tokens = self.tokenizer.tokenize(tokens)
        if trucate_head:
            tokens = tokens[len(tokens) - max_seq_len:]
        else:
            tokens = tokens[:max_seq_len]
        #ids = self._convert_sentence_pair_to_bert_dataset([tokens], max_seq_len)
        if len(tokens) < max_seq_len:
            tokens = tokens + (['[PAD]'] * (max_seq_len - len(tokens)))
        return tokens

    def _convert_sentence_pair_to_bert_dataset(
            self, context, max_len):
        """Convert sentence pairs to dataset for BERT model.

        Args:
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: train: List[int], list of labels
                        test: []

        Returns:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids)
        """
        all_input_ids, all_input_mask, all_segment_ids = [], [], []
        for i, _ in enumerate(context):
            for j, _ in enumerate(context[i]):
                if len(context[i][j]) > max_len:
                    context[i][j] = context[i][j][-max_len:]
                tokens = context[i][j]
                segment_ids = [0] * len(tokens)
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]
                    assert len(tokens) == max_len
                    segment_ids = segment_ids[:max_len]
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                tokens_len = len(input_ids)
                input_ids += [0] * (max_len - tokens_len)
                segment_ids += [0] * (max_len - tokens_len)
                input_mask += [0] * (max_len - tokens_len)
                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_segment_ids.append(segment_ids)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        # test
        return (
            all_input_ids, all_input_mask, all_segment_ids)

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
                q_text = _question['Question']
                temp_question.append(self.convert(content + q_text, self.max_question_len, trucate_head=True))
                for choice in _question["Choices"]:
                    temp_context.append(self.convert(choice, self.max_option_len))
                for _ in range(4 - len(_question["Choices"])):
                    temp_context.append(self.convert("", self.max_option_len))

                context.append(temp_context)
                question.append(temp_question)

        question = self._convert_sentence_pair_to_bert_dataset(question, self.max_question_len)
        context = self._convert_sentence_pair_to_bert_dataset(context, self.max_option_len)
        if mode != "test":
            label = torch.LongTensor(np.array(label, dtype=np.int))
            return {"context": context, "question": question, 'label': label, "id": idx}
        else:
            return {"context": context, "question": question, "id": idx}
