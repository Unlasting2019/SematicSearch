import faiss 
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from train import *
from utils import *

if __name__ == '__main__':
    sup = int(sys.argv[1])
    start = time.time()

    # 读取数据 
    test_dataset = load_data('../../data/test_data', sup)
    train_dataset = load_data('../../data/train_data', sup)
    tokenizer = get_tokenizer(f'{ptm_dir}/vocab.txt')
    a_token, b_token, label = convert_to_ids(test_dataset, tokenizer, maxlen)
    encoder, bert = get_encoder(
        config_path=f'{ptm_dir}/bert_config.json',
        checkpoint_path=f'{ptm_dir}/bert_model.ckpt',
        pooling='first-last-avg',
        dropout_rate=0.1,
        model='bert',
        return_keras_model=False
    )

    # 模型评估
    eval_res = {}
    for i in range(num_epochs):
        bert.load_weights_from_checkpoint(f"model_ckpt/{i}.ckpt")
        a_vec = encoder.predict([a_token, np.zeros_like(a_token)], verbose=True, batch_size=512)
        b_vec = encoder.predict([b_token, np.zeros_like(b_token)], verbose=True, batch_size=512)
        sim = (l2_normalize(a_vec) * l2_normalize(b_vec)).sum(axis=1)
        eval_res[i+1] = compute_corrcoef(label, sim)
    eval_res = sorted(eval_res.items(), key=lambda x : x[1])
    print('eval_res - ', eval_res)

    # 导出向量
    bert.load_weights_from_checkpoint("model_ckpt/{}.ckpt".format(eval_res[-1][0]))
    print('------------ export news_vec -----------')
    neg_topK = 150
    all_text = [_ for _ in open('../../data/news_data').readlines()[:sup]]
    all_token = sequence_padding([tokenizer.encode(_, maxlen=maxlen)[0] for _ in all_text], maxlen)
    all_vec = encoder.predict([all_token, np.zeros_like(all_token)], verbose=True, batch_size=512)
    faiss_index = faiss.index_factory(768, 'Flat', faiss.METRIC_L2)
    faiss_index.add(l2_normalize(all_vec))

    # 生成困难负样本
    print('------------ get hard neg -----------')
    a_token, b_token, label = convert_to_ids(train_dataset, tokenizer, maxlen)
    a_vec = encoder.predict([a_token, np.zeros_like(a_token)], verbose=True, batch_size=512)
    b_vec = encoder.predict([b_token, np.zeros_like(b_token)], verbose=True, batch_size=512)
    a_dist, a_index = faiss_index.search(l2_normalize(a_vec), neg_topK)
    b_dist, b_index = faiss_index.search(l2_normalize(b_vec), neg_topK)

    print('------------ write hard_neg data -----------')
    with open('../../data/hard_neg', 'w+') as f:
        all_data1 = ['{}\t{}\t0'.format(t[0], all_text[i[-1]].strip()) for t, i in zip(train_dataset, a_index)]
        all_data2 = ['{}\t{}\t0'.format(t[1], all_text[i[-1]].strip()) for t, i in zip(train_dataset, b_index)]
        f.write('\n'.join(all_data1+all_data2))

    end = time.time()
    print('-------------- eval done! cost:{}s-------------'.format(round(end-start, 5)))
