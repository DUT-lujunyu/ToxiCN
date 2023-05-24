from transformers import BertModel
# from chinesebert import ChineseBertForMaskedLM, ChineseBertTokenizerFast, ChineseBertConfig
from src.BERT import BertModel
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# 2022.4.28 Glove + LSTM
class BiLSTM(nn.Module):
    def __init__(self, config, embedding_weight):
        super(BiLSTM, self).__init__()
        self.device = config.device
        self.vocab_size = embedding_weight.shape[0]
        self.embed_dim = embedding_weight.shape[1]
        # Embedding Layer
        embedding_weight = torch.from_numpy(embedding_weight).float()        
        embedding_weight = Variable(embedding_weight, requires_grad=config.if_grad)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, _weight=embedding_weight)
        # Encoder layer
        self.bi_lstm = nn.LSTM(self.embed_dim, config.lstm_hidden_dim, bidirectional=True, batch_first=True) 

    def forward(self, **kwargs):
        emb = self.embedding(kwargs["title_text_token_ids"].to(self.device)) # [batch, len] --> [batch, len, embed_dim]
        lstm_out, _ = self.bi_lstm(emb)  # [batch, len, embed_dim] --> [batch, len, lstm_hidden_dim*2]
        lstm_out_pool = torch.mean(lstm_out, dim=1)  # [batch, lstm_hidden_dim*2]
        return lstm_out, lstm_out_pool
    

class Bert_Layer(torch.nn.Module):
    def __init__(self, config):
        super(Bert_Layer, self).__init__()
        # self.use_cuda = kwargs['use_cuda']
        self.device = config.device
        # BERT/Roberta
        self.bert_layer = BertModel.from_pretrained(config.model_name)
        # ChineseBERT
        # self.config = ChineseBertConfig.from_pretrained(config.model_name)
        # self.bert_layer = ChineseBertForMaskedLM.from_pretrained("ShannonAI/ChineseBERT-base", config=self.config)
        self.dim = config.vocab_dim

    def forward(self, **kwargs):
        bert_output = self.bert_layer(input_ids=kwargs['text_idx'].to(self.device),
                                 token_type_ids=kwargs['text_ids'].to(self.device),
                                 attention_mask=kwargs['text_mask'].to(self.device),
                                 toxic_ids=kwargs["toxic_ids"].to(self.device))
        return bert_output[0], bert_output[1]


class TwoLayerFFNNLayer(torch.nn.Module):
    '''
    2-layer FFNN with specified nonlinear function
    must be followed with some kind of prediction layer for actual prediction
    '''
    def __init__(self, config):
        super(TwoLayerFFNNLayer, self).__init__()
        self.output = config.dropout
        self.input_dim = config.vocab_dim
        self.hidden_dim = config.fc_hidden_dim
        self.out_dim = config.num_classes
        self.dropout = nn.Dropout(config.dropout)
        self.model = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                   nn.Tanh(),
                                   nn.Linear(self.hidden_dim, self.out_dim))

    def forward(self, att_input, pooled_emb):
        att_input = self.dropout(att_input)
        pooled_emb = self.dropout(pooled_emb)
        return self.model(pooled_emb)
