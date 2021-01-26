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
import json
import os
from types import SimpleNamespace

import fire
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import Data
from evaluate import evaluate, evaluatetop5
from model import BertForClassification, RnnForSentencePairClassification, LogisticRegression, CharCNN
from utils import load_torch_model

from reshape_loadimage import preprocess

LABELS = ['1', '2']

MODEL_MAP = {
    'bert': BertForClassification,
    'rnn': RnnForSentencePairClassification,
    'lr': LogisticRegression,
    'cnn': CharCNN
}

def main(in_folder='/data/SMP-CAIL2020-test1.csv',
         out_file='/output/result1.json',
         model_config='config/bert_config.json'):
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
    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                max_seq_len=config.max_seq_len,
                model_type=config.model_type, config=config)

    exam_file, filenames = preprocess(in_folder)
    test_set = data.load_file(exam_file, train=False)
    data_loader_test = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False)
    # 2. Load model
    model = MODEL_MAP[config.model_type](config)
    model = load_torch_model(
        model, model_path=os.path.join(config.model_path, 'model.bin'))
    model.to(device)
    # 3. Evaluate
    answer_list = evaluatetop5(model, data_loader_test, device)
    print(answer_list)
    # 4. Write answers to file
    # id_list = pd.read_csv(in_file)['id'].tolist()
    pred_result = dict(zip(filenames,answer_list))
    with open(out_file, 'w') as fout:
        json.dump(pred_result, fout, ensure_ascii=False)




if __name__ == '__main__':
    fire.Fire(main)
