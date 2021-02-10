import argparse
import itertools
import logging
import os
import time
from types import SimpleNamespace
import falcon
import pandas
import torch
from falcon_cors import CORS
import waitress
import numpy as np

import json
import re
from torch.utils.data import DataLoader

from data import Data
from evaluate import evaluate, handy_tool, calculate_accuracy_f1
from model import BERNet, BERXLNet, BERTXLNet, NERNet, NERWNet
from utils import load_torch_model



MODEL_MAP = {
    'bert': BERNet,
    'xlnet': BERXLNet,
    'txlnet': BERTXLNet,
    'rnn': NERNet,
    'rnnkv': NERWNet
}


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
    '-p', '--port', default=58081,
    help='falcon server port')
parser.add_argument(
    '-c', '--config_file', default='config/bert_config-xl.json',
    help='model config file')
args = parser.parse_args()
model_config=args.config_file

# 1)     河流，一种地表水资源名称，如黄河、长江、沙溪、大金川等；
#
# 2)     湖泊，一种地表水资源名称，如太湖、阳澄湖、白洋淀、色林错、洱海等；
#
# 3)     水库，拦洪蓄水和调节水流的水利工程建筑物，如三峡水库、密云水库等；
#
# 4)     水电站，一种水力发电厂，如三峡水电站、葛洲坝水电站等；
#
# 5)     大坝，截河拦水的堤堰，如三峡大坝、丹江口大坝等；
#
# 6)     机构，包括水利管理机构，如“江苏省水利厅”，水利研究机构，如“珠江水利科学研究院”等，水利企业，“山东水利建设集团有限公司”；
#
# 7)     人员，水利相关行政人员名称及水利相关研究人员名称；
#
# 8)     地区，行政区域名称（参考国家行政区划），包括省市区县村的名称；
#
# 9)     水利术语，水利工程、水资源等科研领域的术语，如双曲拱坝、河岸、支流等。

all_types = ['LAK', 'OTH', 'HYD', 'ORG', 'LOC', 'RIV', 'RES', 'TER', 'DAM', 'PER']
cn_all_types = ['湖泊', '其他', '水电站', '机构', '地区', '河流', '水库', '学术术语', '大坝', '人员']

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
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":'na'})
        elif (tag % 3) == 1:
            entity_name += char
            entity_start = idx
        elif (tag % 3) == 2:
            type_index = (tag-1) // 3
            if (entity_name != "") and (i == last):
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": cn_all_types[type_index]})
                entity_name = ""
            else:
                entity_name += char
        elif (tag % 3)+3 == 3:  # or i == len(zipped)
            type_index = (tag-1) // 3
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": cn_all_types[type_index]})
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

    def flatten(self, ll):
        return list(itertools.chain(*ll))

    def cleanall(self, content):
        return content.replace(" ", "", 10**10)

    def split(self, content):
        line = re.findall('(.*?(?:[\n ]|.$))', content)
        sublines = []
        for l in line:
            if len(l) > self.config.max_seq_len:
                ll = re.findall('(.*?(?:[。，]|.$))', l)
                sublines.extend(ll)
            else:
                sublines.append(l)
        sublines = [l for l in sublines if len(l.strip())> 0]
        return sublines

    def bert_classification(self, content):
        logger.info('1:{}'.format(content))
        lines = self.split(content)
        rows = []
        for i,line in enumerate(lines):
            rows.append({"id":i, 'text': line})

        filename = "data/{}.json".format(time.time())
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        test_set, sc_list, label_list = self.data.load_file(filename, train=False)

        token_list = []
        for line in sc_list:
            tokens = self.data.tokenizer.convert_ids_to_tokens(line)
            token_list.append(tokens)

        data_loader_test = DataLoader(
            test_set, batch_size=self.config.batch_size, shuffle=False)

        # 3. Evaluate
        answer_list, length_list = evaluate(self.model, data_loader_test, self.device, isTest=True)

        def flatten(ll):
            return list(itertools.chain(*ll))

        # train_answers = handy_tool(label_list, length_list) #gold
        # #answer_list = handy_tool(answer_list, length_list) #prediction
        # train_answers = flatten(train_answers)
        # train_predictions = flatten(answer_list)
        #
        # train_acc, train_f1 = calculate_accuracy_f1(
        #     train_answers, train_predictions)
        # print(train_acc, train_f1)
        # test_json = json.load(open(config.test_file_path, 'r', encoding='utf-8'))
        # id_list = [item['id'] for item in test_json]

        mod_tokens_list = handy_tool(token_list, length_list)
        result = [result_to_json(t, s) for t, s in zip(mod_tokens_list, answer_list)]
        entity_result = [res["entities"] for res in result]
        return {"data": entity_result}

    def on_get(self, req, resp):
        logger.info("...")
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        content = req.get_param('text', True)
        # clean_content =
        #clean_content = self.cleanall(content)
        resp.media = self.bert_classification(content)
        logger.info("###")


    def on_post(self, req, resp):
        """Handles POST requests"""
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        resp.set_header("Cache-Control", "no-cache")
        data = req.stream.read(req.content_length)
        data = data.decode('utf-8')
        # regex = re.compile(r'\\(?![/u"])')
        # data = regex.sub(r"\\", data)
        jsondata = json.loads(data)
        # clean_title = shortenlines(jsondata['1'])
        # clean_content = cleanall(jsondata['2'])
        content = jsondata['text']
        # clean_content = self.cleanall(content)
        resp.media = self.bert_classification(content)
        logger.info("###")

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/z', TorchResource())
    waitress.serve(api, port=args.port, threads=48, url_scheme='http')
