import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)
import sys
from bert4keras.optimizers import Adam
from utils import *
from bert4keras.snippets import DataGenerator, sequence_padding
from sentence_transformers import SentenceTransformer as SBert
import jieba
jieba.initialize()

ptm_dir = '../data/PTM/paraphrase-multilingual-MiniLM-L12-v2'

if __name__ == '__main__':
    sup = int(sys.argv[1])
    #### 读取数据 ####
    test_data = load_data('data/test_data', sup)
    # 加载模型
    model = SBert(ptm_dir)
    # 模型评估
    a_vec = model.encode([a for a, b, l in test_data])
    b_vec = model.encode([b for a, b, l in test_data])
    label = [l for a, b, l in test_data]
    sim = (l2_normalize(a_vec) * l2_normalize(b_vec)).sum(axis=1)
    corrcoef = compute_corrcoef(label, sim)
    print('corrcoef:', corrcoef)
