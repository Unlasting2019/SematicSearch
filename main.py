import faiss 
import time
import onnxruntime
import os
import sys
sys.path.append('ModelZoo/HardSimCSE/')

from colorama import Fore, Back, Style
from train import *
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]=''

if __name__ == '__main__':
    sup = int(sys.argv[1])
    start = time.time()

    print('---------- load index && model -------------')
    tokenizer = get_tokenizer(f'data/PTM/chinese_L-12_H-768_A-12/vocab.txt')
    all_text = [_ for _ in open('data/news_data').readlines()[:sup*100]]
    faiss_index = faiss.read_index("ModelZoo/HardSimCSE/faiss.index")
    bert_model = onnxruntime.InferenceSession("ModelZoo/HardSimCSE/model.onnx", providers=['CPUExecutionProvider'])

    print('------------ start serving -------------')
    start = time.time()
    inputs = r"""“冬奥体验团”空降吉林！三月的粉雪，滑起来唰唰滴！肖战加盟北京卫视《冬梦之约》，2008年，多少金牌在这里诞生，2022年，冰球之魂将在这里燃起！  肖战加盟《冬梦之约》探索2022年北京冬奥会男子冰球比赛场地国家体育馆，一起为冬奥助力！！3月5日晚21:05，锁定北京卫视！"""
    tokens, segments = tokenizer.encode(inputs, maxlen=maxlen)
    vector = bert_model.run(None, {
            "Input-Token":[tokens],
            "Input-Segment":[segments]
        })
    D, I = faiss_index.search(l2_normalize(vector[0]), 10)

    print(Fore.RED + 'inputs:{}'.format(inputs))
    for i_, (d, i) in enumerate(zip(D[0], I[0])):
        print(Fore.GREEN + 'Top{} - text:{}'.format(i_, all_text[i]))

    end = time.time()
    print('cost:{}s'.format(round(end-start, 6)))
