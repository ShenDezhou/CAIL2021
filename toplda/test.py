import joblib
from gensim.models import ldamodel, CoherenceModel
import pandas as pd

pos_lda = ldamodel.LdaModel.load("model/finmodel.bin")
pos_corpus = joblib.load("model/finmodel.cps")
pos_dict = joblib.load("model/finmodel.dic")


# 正面主题分析
# pos_dict = corpora.Dictionary(po)
# pos_corpus = [pos_dict.doc2bow(i) for i in po]
# pos_lda = ldamulticore.LdaMulticore(pos_corpus,num_topics= 3,id2word =pos_dict, workers=1)
# score_dic = {}
# lda_modes = []
# for n in range(1,5):
#     pos_lda = ldamodel.LdaModel(pos_corpus,num_topics=n * 5,id2word =pos_dict)
#     lda_modes.append(pos_lda)
#     goodcm = CoherenceModel(model=pos_lda, texts=po, dictionary=pos_dict, coherence='c_v', processes=1)
#     score = goodcm.get_coherence()
#     score_dic[n] = score
#     print(score)
#
# Keymax = max(score_dic, key=lambda x: score_dic[x])
# print(Keymax)
# best_top_n = Keymax * 5
# pos_lda = lda_modes[Keymax-1]

# 展示主题
pos_theme = pos_lda.show_topics()


# 取出高频词
import re
pattern = re.compile(r'[\u4e00-\u9fa5]+')
# pattern.findall(pos_theme[0][1])

pos_key_words =[]
for i in range(len(pos_theme)):
    pos_key_words.append(pattern.findall(pos_theme[i][1]))
print(pos_key_words)
pos_key_words = pd.DataFrame(data=pos_key_words)

