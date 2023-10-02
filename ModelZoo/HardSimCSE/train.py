import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE'] = '1'
os.environ['TF_XLA_FLAGS']='--tf_xla_auto_jit=1'
import sys
import time
from utils import *

import tensorflow as tf
#tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)

from bert4keras.backend import keras, K 
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding


ptm_dir = '../../data/PTM/chinese_L-12_H-768_A-12'
batch_size = 18
maxlen = 512
gpu_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
num_epochs = 5

class data_generator(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, token_ids in self.sample(random):
            yield [token_ids[0], token_ids[0], token_ids[1]]

class Evaluator(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        bert.save_weights_as_checkpoint("model_ckpt/{}.ckpt".format(epoch))  

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

def batch_fn(x):
    x_ = {
            'Input-Token':tf.reshape(x, [-1, maxlen]),
            'Input-Segment':tf.reshape(tf.zeros_like(x), [-1, maxlen])
         }
    y_ = {
            'Input-Label':tf.reshape(tf.zeros_like(x), [-1, maxlen])
         }
    return x_, y_

if __name__ == '__main__':
    sup = int(sys.argv[1])
    tokenizer = get_tokenizer(f'{ptm_dir}/vocab.txt')

    #### 读取数据 ####
    start = time.time()
    dataset = load_data('../../data/hard_neg', sup)
    a_token, b_token, lable = convert_to_ids(dataset, tokenizer, maxlen)
    train_dataset = data_generator([(a, b) for a, b in zip(a_token, b_token)], batch_size).to_dataset(
        types='int64',
        shapes=[None, maxlen],
        padded_batch=False,
    ).map(lambda x : batch_fn(x)).prefetch(gpu_num)
    #### 构建模型 ####
    with tf.distribute.MirroredStrategy().scope():
        encoder, bert = get_encoder(
            config_path=f"{ptm_dir}/bert_config.json",
            checkpoint_path=f"{ptm_dir}/bert_model.ckpt",
            pooling='first-last-avg',
            dropout_rate=0.1,
            model='bert',
            return_keras_model=False
        )
        encoder.summary()
        encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))
        bert.load_weights_from_checkpoint(f'{ptm_dir}/bert_model.ckpt') 

    #### 训练模型 ####
    encoder.fit(
        train_dataset,
        steps_per_epoch=len(a_token) // (batch_size * gpu_num),
        epochs=num_epochs,
        callbacks=[Evaluator()]
    )
    end = time.time()
    print('-------------- train done! cost:{}s-------------'.format(round(end-start, 5)))
