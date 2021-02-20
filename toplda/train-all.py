import joblib
import lawa
import pandas as pd
from gensim.models import ldamulticore, ldamodel, CoherenceModel
from gensim import corpora
import pyLDAvis.gensim

positive = pd.read_csv('data/train.csv', encoding = 'utf-8')
# negtive = pd.read_excel(r'C:\Users\Administrator\Desktop\python\项目\爬虫\京东评论\com_neg.xls',encoding = 'utf-8')
# 文本去重(文本去重主要是一些系统自动默认好评的那些评论 )
# positive = positive['content'].drop_duplicates()
# positive = positive['content']
# negtive = negtive ['comment'].drop_duplicates()
# negtive = negtive['comment']


def get_train(positive):
    positive = positive.head(10)
    # 文本分词
    mycut = lambda s: ' '.join(lawa.lcut(s)) # 自定义分词函数
    po =positive.content.apply(mycut)

    # ne =negtive.comment.apply(mycut)

    #停用词过滤（停用词文本可以自己写，一行一个或者用别人整理好的，我这是用别人的）
    # with open(r'C:\Users\Administrator\Desktop\python\项目\电商评论情感分析\stoplist.txt',encoding = 'utf-8') as f:
    #     stop  = f.read()
    # stop =[' ',''] +list(stop[0:])  # 因为读进来的数据缺少空格，我们自己添加进去

    po = po.apply(lambda  s: s.split(' ')) # 将分词后的文本以空格切割
    return po


def get_test(positive):
    positive = positive.head(10)
    # 文本分词
    mycut = lambda s: ' '.join(lawa.lcut(s)) # 自定义分词函数
    po =positive.content.apply(mycut)

    # ne =negtive.comment.apply(mycut)

    #停用词过滤（停用词文本可以自己写，一行一个或者用别人整理好的，我这是用别人的）
    # with open(r'C:\Users\Administrator\Desktop\python\项目\电商评论情感分析\stoplist.txt',encoding = 'utf-8') as f:
    #     stop  = f.read()
    # stop =[' ',''] +list(stop[0:])  # 因为读进来的数据缺少空格，我们自己添加进去

    po = po.apply(lambda  s: s.split(' ')) # 将分词后的文本以空格切割
    return po


def color_words(model, doc):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = True

    # make into bag of words
    doc = model.id2word.doc2bow(doc)
    # get word_topics
    doc_topics, word_topics, phi_values = model.get_document_topics(doc, per_word_topics=True)

    # color-topic matching
    topic_colors = {1: 'red', 0: 'blue'}

    # set up fig to plot
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    # a sort of hack to make sure the words are well spaced out.
    word_pos = 1 / len(doc)

    # use matplotlib to plot words
    for word, topics in word_topics:
        ax.text(word_pos, 0.8, model.id2word[word],
                # font = 'simfang.ttf',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20, color=topic_colors[topics[0]%2],  # choose just the most likely topic
                transform=ax.transAxes)
        word_pos += 0.2  # to move the word for the next iter

    ax.set_axis_off()
    plt.show()

train_dataset = get_train(positive)
#test_dataset = get_test(positive)
# po['2'] = po['1'].apply(lambda x:[i for i in x if i not in stop])# 过滤停用词

# 在这里我们也可以用到之前的词云图分析
# post = []
# for word in po:
#     if len(word)>1 and word not in stop:
#         post.append(word)
# print(post)
# wc = wordcloud.WordCloud(width=1000, font_path='simfang.ttf',height=800)#设定词云画的大小字体，一定要设定字体，否则中文显示不出来
# wc.generate(' '.join(post))
# wc.to_file(r'C:\Users\Administrator\Desktop\python\项目\爬虫\京东评论\yun.png')

# ne['1'] = ne[0:].apply(lambda  s: s.split(' '))
# ne['2'] = ne['1'].apply(lambda x:[i for i in x if i not in stop])


# # 负面主题分析
# neg_dict = corpora.Dictionary(ne['2'])
# neg_corpus = [neg_dict.doc2bow(i) for i in ne['2']]
# neg_lda = ldamulticore.LdaMulticore(neg_corpus,num_topics = 3,id2word = neg_dict, workers=48)

# 正面主题分析
pos_dict = corpora.Dictionary(train_dataset)
pos_corpus = [pos_dict.doc2bow(i) for i in train_dataset]
# pos_lda = ldamulticore.LdaMulticore(pos_corpus,num_topics= 3,id2word =pos_dict, workers=1)
score_dic = {}
lda_modes = []
for n in range(1,5):
    pos_lda = ldamodel.LdaModel(pos_corpus, num_topics=n * 5,id2word =pos_dict)
    lda_modes.append(pos_lda)
    goodcm = CoherenceModel(model=pos_lda, texts=train_dataset, dictionary=pos_dict, coherence='c_v', processes=1)
    score = goodcm.get_coherence()
    score_dic[n] = score
    print(score)

Keymax = max(score_dic, key=lambda x: score_dic[x])
print(Keymax)
best_top_n = Keymax * 5
pos_lda = lda_modes[Keymax-1]

joblib.dump(pos_dict,"model/finmodel.dic", compress=3)
joblib.dump(pos_corpus,"model/finmodel.cps", compress=3)
pos_lda.save("model/finmodel.bin")
# 展示主题
pos_theme = pos_lda.show_topics(num_topics=best_top_n, num_words=30)

all_terms = []
for i in range(best_top_n):
    topic_terms = pos_lda.get_topic_terms(i, 30)
    all_terms.append(topic_terms)

print(all_terms)

# 词主题
law_list = pos_lda.get_term_topics('法规')
case_list = pos_lda.get_term_topics('当事人')
journal_list = pos_lda.get_term_topics('抗拒')
print(law_list, case_list, journal_list)

#文档主题
bow_water = ['工程','甲方','赔偿','合同','工程','甲方','赔偿','合同']
bow = pos_lda.id2word.doc2bow(bow_water) # convert to bag of words format first
doc_topics, word_topics, phi_values = pos_lda.get_document_topics(bow, per_word_topics=True)

print(doc_topics, word_topics)

color_words(pos_lda, bow_water)
# 取出高频词
import re
pattern = re.compile(r'[\u4e00-\u9fa5]+')
# pattern.findall(pos_theme[0][1])

pos_key_words =[]
for i in range(best_top_n):
    pos_key_words.append(pattern.findall(pos_theme[i][1]))
print(pos_key_words)
pos_key_words = pd.DataFrame(data=pos_key_words)



vis = pyLDAvis.gensim.prepare(pos_lda, pos_corpus, pos_dict)
# pyLDAvis==2.1.2
# 在浏览器中心打开一个界面
pyLDAvis.show(vis)