"""Test model for SMP-CAIL2020-Argmine.

Author: Tsinghuaboy tsinghua9boy@sina.com

Usage:
    python main.py --model_config 'config/bert_config.json' \
                   --in_file 'data/SMP-CAIL2020-test1.csv' \
                   --out_file 'bert-submission-test-1.csv'
    python main.py --model_config 'config/rnn_config.json' \
                   --in_file 'data/SMP-CAIL2020-test1.csv' \
                   --out_file 'rnn-submission-test-1.csv'
"""
import argparse
import itertools
import json
import os
import re
from types import SimpleNamespace

import fire
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import Data
from evaluate import evaluate
from model import NERNet,NERWNet, BERNet, BERXLNet, BERTXLNet
from utils import load_torch_model



LABELS = ['1', '2', '3', '4', '5']



MODEL_MAP = {
    'bert': BERNet,
    'xlnet': BERXLNet,
    'txlnet': BERTXLNet,
    'rnn': NERNet,
    'rnnkv': NERWNet
}


all_types = list("ABCDEFGHIJKLMNOPQRSTUVWX")
all_code_dic = {}
with open("data/entity.dic",'r', encoding='utf-8') as f:
    for line in f:
        span = line.strip().split(' ')
        all_code_dic[span[-1]] = span[0]
#
#
# def result_to_json(string, tags):
#     item = {"string": string, "entities": []}
#     entity_name = ""
#     entity_start = 0
#     idx = 0
#     i = -1
#     zipped = zip(string, tags)
#     listzip = list(zipped)
#     last = len(listzip)
#     for char, tag in listzip:
#         i += 1
#         if tag == 0:
#             item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":'s'})
#         elif (tag % 3) == 1:
#             entity_name += char
#             entity_start = idx
#         elif (tag % 3) == 2:
#             type_index = (tag-1) // 3
#             if (entity_name != "") and (i == last):
#                 entity_name += char
#                 item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": all_types[type_index]})
#                 entity_name = ""
#             else:
#                 entity_name += char
#         elif (tag % 3)+3 == 3:  # or i == len(zipped)
#             type_index = (tag-1) // 3
#             entity_name += char
#             item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": all_types[type_index]})
#             entity_name = ""
#         else:
#             entity_name = ""
#             entity_start = idx
#         idx += 1
#     return item
#
# def remove(text):
#     cleanr = re.compile(r"[ !#\$%&'\(\)*\+,-./:;<=>?@\^_`{|}~“”？！【】（）、’‘…￥·]*")
#     cleantext = re.sub(cleanr, '', text)
#     return cleantext

def main(out_file='output/result.json',
         model_config='config/rnn_config.json'):
    """Test model for given test set on 1 GPU or CPU.

    Args:
        in_file: file to be tested
        out_file: output file
        model_config: config file
    """
    # 0. Load config
    with open(model_config) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    # 1. Load data
    data = Data()

    test_set = data.load_user_log(config.test_file_path)

    data_loader_test = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False)
    # 2. Load model
    model = MODEL_MAP[config.model_type](config)
    model = load_torch_model(
        model, model_path=os.path.join(config.model_path, 'model.bin'))
    model.to(device)
    # 3. Evaluate
    answer_list, _ = evaluate(model, data_loader_test, device, isTest=True)

    def flatten(ll):
        return list(itertools.chain(*ll))

    # # 4. Write answers to file
    with open(out_file, 'w', encoding='utf8') as fout:
        for line in answer_list:
            user_profile = []
            for i, e in enumerate(line):
                if e:
                    user_profile.append(all_code_dic[all_types[i]])
            print(user_profile)
            fout.write(",".join(user_profile)+"\n")



if __name__ == '__main__':
    fire.Fire(main)
