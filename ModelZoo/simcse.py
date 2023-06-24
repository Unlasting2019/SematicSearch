import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)
import sys
from bert4keras.optimizers import Adam
from utils import *
from bert4keras.snippets import DataGenerator, sequence_padding
import jieba
jieba.initialize()

ptm_dir = '../data/PTM/chinese_roberta_wwm_ext_L-12_H-768_A-12'

class data_generator(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, token_ids in self.sample(random):
            batch_token_ids.append(token_ids)
            batch_token_ids.append(token_ids)
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
    idxs = K.arange(0, K.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    y_pred = K.l2_normalize(y_pred, axis=1)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)


if __name__ == '__main__':
    sup = int(sys.argv[1])
    tokenizer = get_tokenizer(f'{ptm_dir}/vocab.txt')
    #### 读取数据 ####
    dataset = {
        'train':load_data('data/train_data', sup),
        'test':load_data('data/test_data', sup),
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
    train_generator = data_generator(a_token + b_token, 8)
    encoder.fit(
        train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=5)
    # 模型评估
    a_token, b_token, label = convert_to_ids(dataset['test'], tokenizer, 512)
    a_vec = encoder.predict([a_token, np.zeros_like(a_token)], verbose=True)
    b_vec = encoder.predict([b_token, np.zeros_like(b_token)], verbose=True)
    sim = (l2_normalize(a_vec) * l2_normalize(b_vec)).sum(axis=1)
    corrcoef = compute_corrcoef(label, sim)
    print('corrcoef:', corrcoef)
    # 导出向量
