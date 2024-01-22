# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
from os import path


# 2022.7.27 config基类

class Config_base(object):

    """配置参数"""
    def __init__(self, model_name, dataset):
        # path
        self.model_name = model_name
        self.train_path = path.dirname(path.dirname(__file__))+'/'+ dataset + '/data/train.json'                                # 训练集
        self.dev_path = path.dirname(path.dirname(__file__))+'/'+ dataset + '/data/test.json'                                    # 验证集
        self.test_path = path.dirname(path.dirname(__file__))+'/'+ dataset + '/data/test.json'                                  # 测试集

        self.vocab_path = path.dirname(path.dirname(__file__))+'/'+ dataset + '/data/vocab.pkl' 
        self.lexicon_path = path.dirname(path.dirname(__file__))+'/'+ dataset + '/lexicon/'        # 数据集、模型训练结果                               # 词表
        self.result_path = path.dirname(path.dirname(__file__))+'/' + dataset + '/result'
        self.checkpoint_path = path.dirname(path.dirname(__file__))+'/'+ dataset + '/saved_dict'        # 数据集、模型训练结果
        self.data_path = self.checkpoint_path + '/data.tar'
        # self.log_path = path.dirname(path.dirname(__file__))+'/'+ dataset + '/log/' + self.model_name
        # self.embedding_pretrained = torch.tensor(
        #     np.load(path.dirname(path.dirname(__file__))+'/'+ 'THUCNews' + '/data/' + embedding)["embeddings"].astype('float32'))\
        #     if embedding != 'random' else None                                       # 预训练词向量
        self.word2vec_path = path.dirname(path.dirname(path.dirname(__file__)))+"/glove/source/glove.6B.300d.bin"
        # self.plm = "roberta-base"  # 预训练模型
        # self.plm = "bert-base-chinese"

        # dataset
        self.seed = 1        
        # self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练 transformer:2000
        self.num_classes = 2                                             # 类别数
        # self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.pad_size = 80                                              # 每句话处理成的长度(短填长切)

        # model
        self.dropout = 0.5                                              # 随机失活
        self.vocab_dim = 768
        self.fc_hidden_dim = 256

        # train
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.learning_rate = 1e-5                                       # 学习率  transformer:5e-4 
        self.scheduler = False                                          # 是否学习率衰减
        self.adversarial = False  # 是否对抗训练
        self.num_warm = 0                                               # 开始验证的epoch数
        self.num_epochs = 5                                            # epoch数 
        self.batch_size = 32                                           # mini-batch大小

        # loss
        self.alpha1 = 0.5
        self.gamma1 = 4
        
        # evaluate
        self.threshold = 0.5                                            # 二分类阈值
        self.score_key = "F1"      # 评价指标

if __name__ == '__main__':
    config = Config_base("BERT", "SWSR")
    print(config.vocab_path)
    print(config.word2vec_path)
    print(path.dirname(__file__))
    print(path.dirname(path.dirname(__file__)))