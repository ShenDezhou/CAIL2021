import itertools
import json

import joblib
import lawa
import torch
import pandas
from gensim.models import ldamodel, CoherenceModel
from torch.utils.data import TensorDataset


class Data():

    def __init__(self, lda_model="model/finmodel.bin", lda_dict = "model/finmodel.dic", user_name="data/names.json", property_user="data/result.json"):
        self.pos_dic = joblib.load(lda_dict)
        self.pos_lda = ldamodel.LdaModel.load(lda_model)
        with open(user_name,'r', encoding='utf-8') as f:
            self.user_names = json.load(f)
        self.username_embedding = {}
        for item in self.user_names:
            self.username_embedding[item['text']] = int(item["id"])

        self.all_types = list("ABCDEFGHIJKLMNOPQRSTUVWX")
        with open(property_user, 'r', encoding='utf-8') as f:
            self.user_property = json.load(f)
        self.userproperty_embedding = {}
        for item in self.user_property:
            entities = item["entities"]
            one_hot = self.userproperty_embedding.get(int(item["id"]), torch.zeros((1,len(self.all_types)), device="cuda"))
            for entity in entities:
                type = entity.split('-')[-1]
                one_hot[0, self.all_types.index(type)] += 1
            self.userproperty_embedding[int(item["id"])] = one_hot

        # self.properties = [user['entities'] for user in self.user_property]
        # self.properties = list(set(list(itertools.chain(*self.properties))))

    def load_user_log(self, filename="data/user_search.csv"):
        df = pandas.read_csv(filename)
        topics, properties = [], []
        for row in df.itertuples(index=False):
            words = list(lawa.lcut(row[1]))
            pos_corpus = self.pos_dic.doc2bow(words)
            list_topic = self.pos_lda.get_document_topics(pos_corpus)
            topic = torch.zeros((1,self.pos_lda.num_topics), device='cuda')
            for id, top in list_topic:
                topic[0,id] += top
            user_id = self.username_embedding[row[0]]
            property = self.userproperty_embedding[user_id]
            topics.append(topic)
            properties.append(property)

        topics = torch.cat(topics)
        properties = torch.cat(properties)
        return TensorDataset(topics,  properties)

    def load_train_and_valid_files(self, train_file,valid_file):
        train_data = self.load_user_log(train_file)
        valid_data = self.load_user_log(valid_file)
        return train_data, valid_data




