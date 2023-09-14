import tensorflow as tf
#tf.enable_eager_execution()
import faiss
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)
import sys

from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.layers import *
from utils import *
from sklearn.preprocessing import normalize

import jieba
jieba.initialize()

ptm_dir = '../../data/PTM/chinese_roberta_wwm_ext_L-12_H-768_A-12'

class data_generator(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, token_ids in self.sample(random):
            batch_token_ids.append(token_ids[0])
            batch_token_ids.append(token_ids[0])
            batch_token_ids.append(token_ids[1])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids = []

def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    row = tf.range(0, tf.shape(y_pred)[0], 3)
    idx1 = tf.concat([row, row+1], -1)
    idx2 = tf.concat([row+1, row], -1)
    # 计算相似度
    y_pred = tf.nn.l2_normalize(y_pred, axis=-1)
    sim = tf.einsum("bd,cd->bc", y_pred, y_pred)
    sim = (sim - tf.eye(tf.shape(y_pred)[0]) * 1e12) * 20
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=idx2, logits=tf.gather(sim, idx1))
    return tf.reduce_mean(loss)


if __name__ == '__main__':
    sup = int(sys.argv[1])
    tokenizer = get_tokenizer(f'{ptm_dir}/vocab.txt')
    #### 读取数据 ####
    dataset = {
        'train':load_data('../../data/hard_neg', sup),
        'test':load_data('../../data/test_data', sup),
    }
    a_token, b_token, lable = convert_to_ids(dataset['train'], tokenizer, 512)
    #### 训练模型 ####
    encoder = get_encoder(
        config_path=f'{ptm_dir}/bert_config.json',
        checkpoint_path=f'{ptm_dir}/bert_model.ckpt',
        pooling='first-last-avg',
        dropout_rate=0.1
    )
    # 模型训练
    encoder.summary()
    encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))
    train_generator = data_generator([(a,b) for a,b in zip(a_token,b_token)], 6)
    encoder.fit(
        train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=5)
    encoder.save_weights("HardSimCSE.weights")
    # 模型评估
    a_token, b_token, label = convert_to_ids(dataset['test'], tokenizer, 512)
    a_vec = encoder.predict([a_token, np.zeros_like(a_token)], verbose=True)
    b_vec = encoder.predict([b_token, np.zeros_like(b_token)], verbose=True)
    sim = (l2_normalize(a_vec) * l2_normalize(b_vec)).sum(axis=1)
    corrcoef = compute_corrcoef(label, sim)
    print('corrcoef:', corrcoef)
    encoder.save_weights("encoder_weights")
    print('-------- done! ----------')
