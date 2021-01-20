import argparse
import logging
import os
import re
from types import SimpleNamespace
import falcon
import pandas
import torch
from falcon_cors import CORS
import json
import waitress
from data import Data
from torch.utils.data import DataLoader
from utils import load_torch_model
from model import NERNet,NERWNet, BERNet, BERXLNet, BERTXLNet
from evaluate import evaluate, handy_tool
import time
from dataclean import cleanall, shortenlines

logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
logger = logging.getLogger()
cors_allow_all = CORS(allow_all_origins=True,
                      allow_origins_list=['*'],
                      allow_all_headers=True,
                      allow_all_methods=True,
                      allow_credentials_all_origins=True
                      )

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config_file', default='config/rnn_config.json',
    help='model config file')
args = parser.parse_args()
model_config=args.config_file

MODEL_MAP = {
    'bert': BERNet,
    'xlnet': BERXLNet,
    'txlnet': BERTXLNet,
    'rnn': NERNet,
    'rnnkv': NERWNet
}


all_types = ['TOT', 'PHA', 'RES']

def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    i = -1
    zipped = zip(string, tags)
    listzip = list(zipped)
    last = len(listzip)
    for char, tag in listzip:
        i += 1
        if tag == 0:
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":'s'})
        elif (tag % 3) == 1:
            entity_name += char
            entity_start = idx
        elif (tag % 3) == 2:
            type_index = (tag-1) // 3
            if (entity_name != "") and (i == last):
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": all_types[type_index]})
                entity_name = ""
            else:
                entity_name += char
        elif (tag % 3)+3 == 3:  # or i == len(zipped)
            type_index = (tag-1) // 3
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": all_types[type_index]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item

class TorchResource:

    def __init__(self):
        logger.info("...")
        # 0. Load config
        with open(model_config) as fin:
            self.config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.max_seq_len = self.config.max_seq_len
        # 1. Load data
        self.data = Data(vocab_file=os.path.join(self.config.model_path, 'vocab.txt'),
                    max_seq_len=self.config.max_seq_len,
                    model_type=self.config.model_type, config=self.config)

        # 2. Load model
        self.model = MODEL_MAP[self.config.model_type](self.config)
        self.model = load_torch_model(
            self.model, model_path=os.path.join(self.config.model_path, 'model.bin'))
        self.model.to(self.device)

        logger.info("###")


    def split(self, content):
        line = re.findall('(.*?(?:[\n。，]|.$))', content)
        sublines = []
        for l in line:
            if len(l) > self.max_seq_len:
                ll = re.findall('(.*?(?:[；？！]|.$))', l)
                sublines.extend(ll)
            else:
                sublines.append(l)
        sublines = [l for l in sublines if len(l.strip())> 0]
        return sublines

    def bert_classification(self, content):
        logger.info('1:{}'.format(content))
        # row = {'type1': '/', 'title': title, 'content': content}
        # df = pandas.DataFrame().append(row, ignore_index=True)
        filename = "data/{}.csv".format(time.time())
        lines = self.split(content)
        items = [{"text":line} for line in lines]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=4)
        # df.to_csv(filename, index=False, columns=['type1', 'title', 'content'])
        test_set, sc_list, label_list = self.data.load_file(filename, train=False)

        data_loader_test = DataLoader(
            test_set, batch_size=self.config.batch_size, shuffle=False)
        # Evaluate
        answer_list, length_list = evaluate(self.model, data_loader_test, self.device, isTest=True)

        token_list = []
        for line in sc_list:
            tokens = self.data.tokenizer.convert_ids_to_tokens(line)
            token_list.append(tokens)

        mod_tokens_list = handy_tool(token_list, length_list)
        result = [result_to_json(t, s) for t, s in zip(mod_tokens_list, answer_list)]
        entity_list = []
        for item in result:
            entities = item['entities']
            words = [d['word']+"-"+d['type'] for d in entities if d['type'] !='s']
            entity_list.extend(words)
        return {"answer": entity_list}

    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        content = req.get_param('1', True)
        # content = req.get_param('2', True)
        # clean_title = shortenlines(title)
        clean_content = cleanall(content)
        resp.media = self.bert_classification(clean_content)
        logger.info("###")


    def on_post(self, req, resp):
        """Handles POST requests"""
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        resp.set_header("Cache-Control", "no-cache")
        data = req.stream.read(req.content_length)
        jsondata = json.loads(data)
        # clean_title = shortenlines(jsondata.title)
        clean_content = cleanall(jsondata['1'])
        resp.media = self.bert_classification(clean_content)

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=58080, threads=48, url_scheme='http')
