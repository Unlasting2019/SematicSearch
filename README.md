# SematicSearch
毕业设计项目，在有标注的搜狐的新闻文本匹配数据集(数据集格式为q1, q2, label)上进行算法实验，通过无监督模型得到句子的Embedding，样本之间的Spearman系数作为评测指标

## 算法部分
- baseline（backbone:bert - pooling:cls）:0.2872
- SimCSE:0.3924
- SimCSE+困难负样本（参考FaceBook的实现）:0.4488
