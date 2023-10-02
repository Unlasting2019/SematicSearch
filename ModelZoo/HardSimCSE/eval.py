import faiss 
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from train import *
from utils import *
import tf2onnx

if __name__ == '__main__':
    sup = int(sys.argv[1])
    start = time.time()

    # 读取数据 
    dataset = load_data('../../data/test_data', sup)
    tokenizer = get_tokenizer(f'{ptm_dir}/vocab.txt')
    a_token, b_token, label = convert_to_ids(dataset, tokenizer, maxlen)
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
    all_text = [_.strip() for _ in open('../../data/news_data').readlines()[:sup]]
    all_token = sequence_padding([tokenizer.encode(_, maxlen=maxlen)[0] for _ in all_text], maxlen)
    all_vec = encoder.predict([all_token, np.zeros_like(all_token)], verbose=True, batch_size=512)
    all_norm_vec = l2_normalize(all_vec)
    faiss_index = faiss.index_factory(768, 'Flat', faiss.METRIC_L2)
    faiss_index.add(all_vec)
    faiss.write_index(faiss_index, 'faiss.index')
    pd.DataFrame({'news_data':all_text, 'vector_data':[';'.join(_.astype(str)) for _ in all_norm_vec]}).to_csv('../../data/news_vector', sep='\t', index=False, header=None)

    # 导出模型
    tf2onnx.convert.from_keras(encoder, output_path="model.onnx")
