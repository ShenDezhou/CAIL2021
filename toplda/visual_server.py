import argparse

import joblib
import pyLDAvis.gensim
from gensim.models import ldamodel


parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--ip', default="0.0.0.0",
    help='falcon server ip')
parser.add_argument(
    '-p', '--port', default=58063,
    help='falcon server port')
parser.add_argument(
    '-c', '--config_file', default='model/finmodel',
    help='model config file')
args = parser.parse_args()

if __name__=="__main__":
    pos_lda = ldamodel.LdaModel.load(args.config_file + ".bin")
    pos_corpus = joblib.load(args.config_file + ".cps")
    pos_dict = joblib.load(args.config_file + ".dic")

    vis = pyLDAvis.gensim.prepare(pos_lda, pos_corpus, pos_dict)
    # pyLDAvis==2.1.2
    # 在浏览器中心打开一个界面
    pyLDAvis.show(vis, ip=args.ip, port=int(args.port))