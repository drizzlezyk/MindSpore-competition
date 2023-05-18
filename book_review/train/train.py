#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module providing baseline"""

import re
import csv
import os
import argparse
import numpy as np

import mindspore.dataset as ds
from mindspore import context

from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore import set_context, PYNATIVE_MODE, Model, nn
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor

from rnn import RNN


def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(description="train lstm",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--pretrain_url', type=str,
                        default='./result/test.ckpt',
                        help='the pretrain model path')

    parser.add_argument('--data_path',
                        type=str, default='./train.csv',
                        help='imbd path')

    parser.add_argument('--test_path',
                        type=str, default='./test.csv',
                        help='imbd path')

    parser.add_argument('--glove_path',
                        type=str,
                        default='./glove.6B.100d.txt',
                        help='glove path')

    parser.add_argument('--cache_dir',
                        type=str,
                        default='C:\\datasets\\',
                        help='glove path')

    parser.add_argument('--output_path',
                        default='./result',
                        type=str, help='the path model saved')

    parser.add_argument('--epochs',
                        default=2,
                        type=int,
                        help='training epochs')

    parser.add_argument('--lr',
                        default=0.0001,
                        type=float,
                        help='learning rate')

    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='batch size')
    return parser.parse_args()


class AmazonData():
    """
    文本CSV数据集加载器
    加载数据集并处理为一个Python迭代对象。

    """

    def __init__(self, path):
        self.path = path
        self.review, self.label = [], []
        self._load()

    def _load(self):
        # 根据self.path load 需要读的csv文件
        with open(self.path, "r") as csv_file:
            dict_reader = csv.DictReader(csv_file)

            # 按行读取
            for row in dict_reader:
                review = row['review']
                label = int(np.float32(row['label']))
                # 数据处理
                label_onehot = [0] * 5
                label_onehot[label - 1] = 1
                review = re.split(' |,', review.lower())
                # 数据加载到List
                self.review.append(review)
                self.label.append(label_onehot)

    def __getitem__(self, idx):
        """
        定义可迭代对象返回当前结果的逻辑
        """
        return self.review[idx], self.label[idx]

    def __len__(self):
        """
        返回可迭代对象的长度
        :return: int
        """
        return len(self.review)


def load_glove():
    glove_100d_path = args.glove_path
    embeddings = []
    tokens = []
    with open(glove_100d_path, encoding='utf-8') as file:
        for glove in file:
            word, embedding = glove.split(maxsplit=1)
            tokens.append(word)
            embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))
    # 添加 <unk>, <pad> 两个特殊占位符对应的embedding
    embeddings.append(np.random.rand(100))
    embeddings.append(np.zeros((100,), np.float32))

    vocab = ds.text.Vocab.from_list(tokens, special_tokens=["<unk>", "<pad>"], special_first=False)
    embeddings = np.array(embeddings).astype(np.float32)
    return vocab, embeddings


def load_process_data():
    data_train = ds.GeneratorDataset(AmazonData(args.data_path),
                                     column_names=["review", "label"])
    data = next(data_train.create_dict_iterator())
    print(data["review"])
    print(data["label"])

    set_context(mode=PYNATIVE_MODE)

    idx = vocab.tokens_to_ids('the')
    embedding = embeddings[idx]
    print('the: ', embedding)
    batch_size = args.batch_size

    lookup_op = ds.text.Lookup(vocab, unknown_token='<unk>')
    pad_op = ds.transforms.c_transforms.PadEnd([100], pad_value=vocab.tokens_to_ids('<pad>'))

    data = next(data_train.create_dict_iterator())
    print(data["review"])
    print(type(data["label"]))

    data_train = data_train.map(operations=[lookup_op, pad_op], input_columns=['review'])

    data_train, data_valid = data_train.split([0.7, 0.3])

    data_train = data_train.batch(batch_size, drop_remainder=True)
    data_valid = data_valid.batch(batch_size, drop_remainder=True)

    data = next(data_train.create_dict_iterator())

    print(data["review"])
    print(data["label"])

    print(len(data))
    return data_train, data_valid


def train():
    learning_rate = 0.001

    network = RNN(embeddings,
                  hidden_dim=hidden_dim,
                  output_dim=output_dim,
                  n_layers=n_layers,
                  bidirectional=bidirectional,
                  dropout=dropout,
                  pad_idx=vocab.tokens_to_ids('<pad>'))

    loss = nn.MSELoss(reduction='mean')
    optimizer = nn.Adam(network.trainable_params(), learning_rate=learning_rate)

    model = Model(network, loss, optimizer, metrics={"Accuracy": nn.Accuracy()})

    num_epochs = args.epochs
    loss_monitor = LossMonitor(50)

    model.train(num_epochs,
                data_train,

                callbacks=loss_monitor)
    save_checkpoint(network, os.path.join(args.output_path, "test.ckpt"))


def predict_sentiment(model, sentence):
    model.set_train(False)
    prediction = model(sentence)
    return prediction


def test_predict():
    data_test = ds.GeneratorDataset(AmazonData(args.test_path),
                                    column_names=["review", "label"])

    lookup_op = ds.text.Lookup(vocab, unknown_token='<unk>')
    pad_op = ds.transforms.c_transforms.PadEnd([100], pad_value=vocab.tokens_to_ids('<pad>'))
    data_test = data_test.map(operations=[lookup_op, pad_op], input_columns=['review'])
    data_test = data_test.batch(1, drop_remainder=True)

    ckpt_file_name = args.pretrain_url
    network = RNN(embeddings,
                  hidden_dim=hidden_dim,
                  output_dim=output_dim,
                  n_layers=n_layers,
                  bidirectional=bidirectional,
                  dropout=dropout,
                  pad_idx=vocab.tokens_to_ids('<pad>'))
    param_dict = load_checkpoint(ckpt_file_name)
    load_param_into_net(network, param_dict)

    res = []

    for row in data_test.create_dict_iterator():
        review = row['review']
        result = np.argmax(predict_sentiment(network, review)) + 1

        res.append(result)

    print(res)
    with open(os.path.join(args.output_path, 'result.txt'), "w") as f:
        for r in res:
            f.write(str(r) + '\n')


if __name__ == '__main__':
    args = parse_args()
    hidden_dim = 256
    output_dim = 5
    n_layers = 2
    bidirectional = True
    dropout = 0.5
    vocab, embeddings = load_glove()
    data_train, data_valid = load_process_data()
    train()
    # test_predict()


