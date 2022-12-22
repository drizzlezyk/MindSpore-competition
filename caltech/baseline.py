#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module providing baseline"""

import json
from typing import Type, Union, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as transforms

from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.head import DenseHead
from mindvision.classification.models.neck import GlobalAvgPooling
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel
from mindvision.engine.callback import ValAccMonitor
from mindvision.classification.models.blocks import ConvNormActivation

from mindspore import nn, load_checkpoint, load_param_into_net, Tensor
from mindspore.train import Model


# -------------------------------- building block -------------------------------------------
class ResidualBlockBase(nn.Cell):
    expansion: int = 1  # 最后一个卷积核数量与第一个卷积核数量相等

    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None) -> None:
        super().__init__()
        if not norm:
            norm = nn.BatchNorm2d

        self.conv1 = ConvNormActivation(in_channel, out_channel,
                                        kernel_size=3, stride=stride, norm=norm)
        self.conv2 = ConvNormActivation(out_channel, out_channel,
                                        kernel_size=3, norm=norm, activation=None)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        """ResidualBlockBase construct."""
        identity = x  # shortcuts分支

        out = self.conv1(x)  # 主分支第一层：3*3卷积层
        out = self.conv2(out)  # 主分支第二层：3*3卷积层

        if self.down_sample:
            identity = self.down_sample(x)
        out += identity  # 输出为主分支与shortcuts之和
        out = self.relu(out)

        return out


# --------------------------- Bottleneck --------------------------------------
class ResidualBlock(nn.Cell):
    expansion = 4  # 最后一个卷积核的数量是第一个卷积核数量的4倍

    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None) -> None:
        super(ResidualBlock, self).__init__()
        if not norm:
            norm = nn.BatchNorm2d

        self.conv1 = ConvNormActivation(in_channel, out_channel,
                                        kernel_size=1, norm=norm)
        self.conv2 = ConvNormActivation(out_channel, out_channel,
                                        kernel_size=3, stride=stride, norm=norm)
        self.conv3 = ConvNormActivation(out_channel, out_channel * self.expansion,
                                        kernel_size=1, norm=norm, activation=None)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        identity = x  # shortscuts分支

        out = self.conv1(x)  # 主分支第一层：1*1卷积层
        out = self.conv2(out)  # 主分支第二层：3*3卷积层
        out = self.conv3(out)  # 主分支第三层：1*1卷积层

        if self.down_sample:
            identity = self.down_sample(x)

        out += identity  # 输出为主分支与shortcuts之和
        out = self.relu(out)

        return out


def make_layer(last_out_channel, block: Type[Union[ResidualBlockBase, ResidualBlock]],
               channel: int, block_nums: int, stride: int = 1):
    down_sample = None  # shortcuts分支

    if stride != 1 or last_out_channel != channel * block.expansion:
        down_sample = ConvNormActivation(last_out_channel,
                                         channel * block.expansion,
                                         kernel_size=1,
                                         stride=stride,
                                         norm=nn.BatchNorm2d,
                                         activation=None)

    layers = []
    layers.append(block(last_out_channel,
                        channel,
                        stride=stride,
                        down_sample=down_sample,
                        norm=nn.BatchNorm2d))

    in_channel = channel * block.expansion
    # 堆叠残差网络
    for _ in range(1, block_nums):
        layers.append(block(in_channel,
                            channel,
                            norm=nn.BatchNorm2d))

    return nn.SequentialCell(layers)


class ResNet(nn.Cell):
    def __init__(self, block: Type[Union[ResidualBlockBase, ResidualBlock]],
                 layer_nums: List[int], norm: Optional[nn.Cell] = None) -> None:
        super(ResNet, self).__init__()
        if not norm:
            norm = nn.BatchNorm2d
        # 第一个卷积层，输入channel为3（彩色图像），输出channel为64
        self.conv1 = ConvNormActivation(3, 64, kernel_size=7, stride=2, norm=norm)
        # 最大池化层，缩小图片的尺寸
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        # 各个残差网络结构块定义，
        self.layer1 = make_layer(64, block, 64, layer_nums[0])
        self.layer2 = make_layer(64 * block.expansion, block, 128, layer_nums[1], stride=2)
        self.layer3 = make_layer(128 * block.expansion, block, 256, layer_nums[2], stride=2)
        self.layer4 = make_layer(256 * block.expansion, block, 512, layer_nums[3], stride=2)

    def construct(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def _resnet(arch: str, block: Type[Union[ResidualBlockBase, ResidualBlock]],
            layers: List[int], num_classes: int, pretrained: bool, input_channel: int):
    backbone = ResNet(block, layers)
    neck = GlobalAvgPooling()  # 平均池化层
    head = DenseHead(input_channel=input_channel, num_classes=num_classes)  # 全连接层
    model = BaseClassifier(backbone, neck, head)  # 将backbone层、neck层和head层连接起来

    if pretrained:
        # 下载并加载预训练模型
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model


def resnet50(num_classes: int = 1000, pretrained: bool = False):
    "ResNet50模型"
    return _resnet("resnet50", ResidualBlock, [3, 4, 6, 3], num_classes, pretrained, 2048)


def load_label_dict():
    with open("./dataset/label_dict.json", "r") as file:
        for line in file:
            label_dict = json.loads(line)
        print(label_dict)
    return label_dict


def load_data(train_dir, test_dir):
    image_size = 32
    mean = [0.5 * 255] * 3
    std = [0.5 * 255] * 3
    trans = [
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=mean, std=std),
        transforms.HWC2CHW()
    ]

    ds_train = ds.ImageFolderDataset(train_dir, decode=True)
    ds_train = ds_train.map(operations=trans, num_parallel_workers=1)
    ds_test = ds.ImageFolderDataset(test_dir, decode=True)
    ds_test = ds_test.map(operations=trans, num_parallel_workers=1)

    ds_train, ds_val = ds_train.split([0.8, 0.2])

    batch_size = 512
    ds_train = ds_train.batch(batch_size, drop_remainder=True)
    ds_val = ds_val.batch(batch_size, drop_remainder=True)

    for _ in range(10):
        data = next(ds_train.create_dict_iterator())
        print(data['label'])
        print(data['image'].shape)

    return ds_train, ds_val, ds_test


def train(ds_train, ds_val):
    # 定义ResNet50网络
    network = resnet50(pretrained=False)
    param_dict = load_checkpoint("C:\\Users\\Administrator\\"
                                 "PycharmProjects\\ms_tutorial\\model_save\\resnet50_224.ckpt")
    load_param_into_net(network, param_dict)

    # 全连接层输入层的大小
    in_channel = network.head.dense.in_channels
    # 重置全连接层
    network.head = DenseHead(input_channel=in_channel, num_classes=256)
    # 设置学习率
    num_epochs = 1
    step_size = ds_train.get_dataset_size()
    learning_rate = nn.cosine_decay_lr(min_lr=0.00001,
                            max_lr=0.001,
                            total_step=step_size * num_epochs,
                            step_per_epoch=step_size,
                            decay_epoch=num_epochs)
    # 定义优化器和损失函数
    opt = nn.Momentum(params=network.trainable_params(),
                      learning_rate=learning_rate,
                      momentum=0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True,
                                            reduction='mean')
    # 实例化模型
    model = Model(network,
                  loss,
                  opt,
                  metrics={"Accuracy": nn.Accuracy()})

    print('image_classification training')
    # 模型训练
    model.train(num_epochs,
                ds_train,
                callbacks=[
                    ValAccMonitor(model,
                                  ds_val,
                                  num_epochs,
                                  ckpt_directory='./model/resnet50')
                ])

    print('finish training')


def visualize_model(best_ckpt_path, val_ds):
    label_dict = load_label_dict()
    net = resnet50(256)

    # 加载模型参数
    param_dict = load_checkpoint(best_ckpt_path)
    load_param_into_net(net, param_dict)
    model = Model(net)

    # 加载验证集的数据进行验证
    data = next(val_ds.create_dict_iterator())
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()

    # 预测图像类别
    output = model.predict(Tensor(data['image']))
    pred = np.argmax(output.asnumpy(), axis=1)

    with open('./result/result_example.txt', 'w') as file:
        for result in pred:
            print(result)
            file.writelines(str(result)+'\n')

    # 显示图像及图像的预测值
    plt.figure()
    for i in range(1, 7):
        plt.subplot(2, 3, i)

        # 若预测正确，显示为蓝色；若预测错误，显示为红色
        color = 'blue' if pred[i - 1] == labels[i - 1] else 'red'
        val_ds.index2label = label_dict
        plt.title('predict:{}'.format(label_dict[str(pred[i - 1])]), color=color)

        picture_show = np.transpose(images[i - 1], (1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        picture_show = std * picture_show + mean
        picture_show = np.clip(picture_show, 0, 1)

        plt.imshow(picture_show)
        plt.axis('off')

    plt.show()


def run_baseline():
    dataset_dir = "C:\\datasets\\caltech_for_user_final\\train"
    test_dir = "C:\\datasets\\caltech_for_user_final\\test"

    ds_train, ds_val, ds_test = load_data(dataset_dir, test_dir)
    train(ds_train, ds_val)

    # 使用测试数据集进行验证
    visualize_model('model/resnet50/best.ckpt', ds_test)
