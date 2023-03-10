#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module of RNN"""

import math
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import Uniform, HeUniform


class RNN(nn.Cell):
    def __init__(self, embeddings, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      embedding_table=Tensor(embeddings),
                                      padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
        self.predict = nn.Dense(hidden_dim * 2,
                                output_dim,
                                weight_init=weight_init,
                                bias_init=bias_init)
        self.dropout = nn.Dropout(1 - dropout)
        self.softmax = ops.Softmax()

    def construct(self, inputs):
        """
        :param inputs:
        :return:
        """
        embedded = self.dropout(self.embedding(inputs))
        _, (hidden, _) = self.rnn(embedded)
        hidden = self.dropout(mnp.concatenate((hidden[-2, :, :], hidden[-1, :, :]), axis=1))
        output = self.predict(hidden)
        return self.softmax(output)
