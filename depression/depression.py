import re
import csv
import os
import argparse
import numpy as np

import mindspore
import mindspore.dataset as ds

from mindspore import load_checkpoint, load_param_into_net
from mindspore import set_context, PYNATIVE_MODE, Model, nn
from mindvision.engine.callback import ValAccMonitor

from rnn import RNN

CACHE_DIR = "./"


def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(description="train lstm",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--pretrain_url', type=str,
                        default='./resnet_01.ckpt',
                        help='the pretrain model path')

    parser.add_argument('--imdb_path',
                        type=str, default='./aclImdb_v1.tar.gz',
                        help='imbd path')

    parser.add_argument('--glove_path',
                        type=str,
                        default='./glove.6B.zip',
                        help='glove path')

    parser.add_argument('--cache_dir',
                        type=str,
                        default='./',
                        help='glove path')

    parser.add_argument('--out_path',
                        default='./save_model/',
                        type=str, help='the path model saved')

    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help='training epochs')

    parser.add_argument('--lr',
                        default=0.0001,
                        type=float,
                        help='learning rate')

    return parser.parse_args(args=[])


class depression():
    """Amazon数据集加载器

    加载Amazon数据集并处理为一个Python迭代对象。

    """

    def __init__(self, path):
        self.path = path
        self.review, self.label = [], []

        self._load()

    def _load(self):
        with open(self.path, "r") as csv_file:
            dict_reader = csv.DictReader(csv_file)

            for row in dict_reader:
                review = row['clean_text']
                label = int(np.float32(row['is_depression']))
                label_onehot = [0] * 2
                label_onehot[label-1] = 1
                # review = str(review.translate(None, six.b(string.punctuation))).split())
                review = re.split(' |,', review.lower())
                self.review.append(review)
                self.label.append(label_onehot)

    def __getitem__(self, idx):
        return self.review[idx], self.label[idx]

    def __len__(self):
        return len(self.review)


def load_glove():
    glove_100d_path = os.path.join(CACHE_DIR, 'glove.6B.100d.txt')
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


def load_data(data_path="./datasets/train.csv"):
    dataset = ds.CSVDataset(dataset_files=data_path, column_names=["clean_text", "is_depression"])
    data = next(dataset.create_dict_iterator())
    print(data["clean_text"])
    return dataset


def load_process_data():
    data_train = ds.GeneratorDataset(depression("./datasets/train.csv"),
                                     column_names=["clean_text", "is_depression"])
    data = next(data_train.create_dict_iterator())
    print(data["clean_text"])
    print(data["is_depression"])
    vocab, embeddings = load_glove()

    set_context(mode=PYNATIVE_MODE)

    idx = vocab.tokens_to_ids('the')
    embedding = embeddings[idx]
    print('the: ', embedding)
    BATCH_SIZE = 1

    lookup_op = ds.text.Lookup(vocab, unknown_token='<unk>')
    pad_op = ds.transforms.c_transforms.PadEnd([400], pad_value=vocab.tokens_to_ids('<pad>'))

    data = next(data_train.create_dict_iterator())
    print(type(data["clean_text"]))
    print(type(data["is_depression"]))

    data_train = ds.GeneratorDataset(data_train, column_names=["clean_text", "is_depression"])
    data_train = data_train.map(operations=[lookup_op, pad_op], input_columns=['clean_text'])

    data_train, data_valid = data_train.split([0.7, 0.3])

    data_train = data_train.batch(BATCH_SIZE, drop_remainder=True)
    data_valid = data_valid.batch(BATCH_SIZE, drop_remainder=True)

    data = next(data_train.create_dict_iterator())

    print(data["clean_text"])
    print(data["is_depression"])

    print(len(data))
    return data_train, data_valid, vocab, embeddings


def train(data_train, data_valid, vocab, embeddings):
    learning_rate = 0.001

    network = RNN(embeddings,
                  hidden_dim=256,
                  output_dim=2,
                  n_layers=5,
                  bidirectional=True,
                  dropout=0.5,
                  pad_idx=vocab.tokens_to_ids('<pad>'))

    loss = nn.MSELoss(reduction='mean')
    optimizer = nn.Adam(network.trainable_params(), learning_rate=learning_rate)

    model = Model(network, loss, optimizer, metrics={"Accuracy": nn.Accuracy()})

    num_epochs = 1
    model.train(num_epochs,
                data_train,
                callbacks=[
                    ValAccMonitor(model, data_valid, num_epochs, ckpt_directory='./model/')
                ])


def predict_sentiment(model, vocab, sentence):
    model.set_train(False)
    tokenized = re.split(' |,.', sentence.lower())
    indexed = vocab.tokens_to_ids(tokenized)
    tensor = mindspore.Tensor(indexed, mindspore.int32)
    tensor = tensor.expand_dims(0)
    if len(tokenized) == 1:
        tensor = tensor.expand_dims(0)
    prediction = model(tensor)
    return prediction


def test_predict(vocab, embeddings):

    ckpt_file_name = "./model/best.ckpt"
    network = RNN(embeddings,
                  hidden_dim=256,
                  output_dim=2,
                  n_layers=5,
                  bidirectional=True,
                  dropout=0.5,
                  pad_idx=vocab.tokens_to_ids('<pad>'))
    param_dict = load_checkpoint(ckpt_file_name)
    load_param_into_net(network, param_dict)
    res = []
    with open("./datasets/test.csv", "r") as test_file:
        test_dict_reader = csv.DictReader(test_file)
        for row in test_dict_reader:
            review = row['clean_text']
            result = np.argmax(predict_sentiment(network, vocab, review))
            res.append(result)
    print(res)
    

if __name__ == "__main__":
    data_train, data_valid, vocab, embeddings = load_process_data()
    train(data_train, data_valid, vocab, embeddings)
    test_predict(vocab, embeddings)
