#_*_ coding:utf-8 _*_

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet(nn.Module):
    """attention!

    这个版本的网络将原始网络缩小了8倍！
    网络最开始有一个卷积层，之后跟着四个block，每一个block内部都是密集连接的
    在每两个block的中间有一个Transition_layer：过度层
    这个版本的网络是原文中表现最好的DenseNet-BC结构，也就是通过过渡层和bottleneck结构进行降维的设计

    DenseNet-121:6,12,24,16
    DenseNet-169:6,12,32,32
    DenseNet-201:6,12,48,32
    DenseNet-264:6,12,64,48

    原始论文中所有卷积层的通道数量都是受到增加率的影响的,并且对于imagenet的增加率设置为了32
    """
    def __init__(self, block_nums, growth_rate, color_channels, class_nums):
        """

        :param block_nums: 每一个stage中包含的block的数量
        :param growth_rate: 论文中的名词——增长率
        :param color_channels: 原始图像数据中的通道数量
        :param class_nums: 一共要分成多少个类别
        """
        super(DenseNet, self).__init__()

        self.num_feature = growth_rate * 2  # 原始论文中第一个卷积层的个数是增长率的两倍

        self.growth_rate = growth_rate

        self.conv1 = nn.Sequential(  # 第一层还是需要根据实际问题来进行调整的
            nn.Conv2d(color_channels, self.num_feature, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.num_feature),
            nn.ReLU()

            #nn.MaxPool2d(3, 2, 1)
        )

        self.block1 = nn.Sequential(
            self.make_block(block_nums[0]),
            Transition_layer(self.num_feature)
        )

        self.num_feature = self.num_feature // 2

        self.block2 = nn.Sequential(
            self.make_block(block_nums[1]),
            Transition_layer(self.num_feature)
        )

        self.num_feature = self.num_feature // 2

        self.block3 = nn.Sequential(
            self.make_block(block_nums[2]),
            Transition_layer(self.num_feature)
        )

        self.num_feature = self.num_feature // 2

        self.block4 = nn.Sequential(
            self.make_block(block_nums[3]),
            nn.AvgPool2d(4)  # 是需要根据具体的图像大小来进行调整的
        )

        self.final_bn = nn.Sequential(
            nn.BatchNorm2d(self.num_feature),
            nn.ReLU()
        )

        self.clf = nn.Sequential(
            nn.Linear(self.num_feature, class_nums),
            nn.Softmax()
        )

        # 参数初始化

        for layer in self.modules():

            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal(layer.weight.data)

            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

            elif isinstance(layer, nn.Linear):
                layer.bias.data.zero_()

    def forward(self, inputs):

        outputs = self.conv1(inputs)

        outputs = self.block1(outputs)
        outputs = self.block2(outputs)
        outputs = self.block3(outputs)
        outputs = self.block4(outputs)

        outputs = self.final_bn(outputs)

        outputs = outputs.view(outputs.size(0), -1)

        return self.clf(outputs)

    def make_block(self, blcok_nums):

        layer = []

        for i in range(blcok_nums):

            layer += [Convlayer(self.num_feature, self.growth_rate)]
            self.num_feature += self.growth_rate

        return nn.Sequential(*layer)


class Convlayer(nn.Module):  # 每一个block中的一个卷积层由两个卷积组成，一个是1*1，另外一个是3*3

    def __init__(self, inchannels, growth_rate, bn_size=4):
        super(Convlayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(inchannels)
        self.conv1 = nn.Conv2d(inchannels, growth_rate * bn_size, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(growth_rate * bn_size)
        self.conv2 = nn.Conv2d(growth_rate * bn_size, growth_rate, 3, 1, 1, bias=False)

    def forward(self, inputs):

        outputs = self.bn1(inputs)
        outputs = F.relu(outputs)
        outputs = self.conv1(outputs)

        outputs = self.bn2(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)

        return torch.cat((inputs, outputs), dim=1)


class Transition_layer(nn.Module):  # 每两个block中间的过度层,过度层会将通道的维度降低一倍

    def __init__(self, inchannels):
        super(Transition_layer, self).__init__()

        self.bn = nn.BatchNorm2d(inchannels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inchannels, inchannels // 2, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, inputs):
        return self.pool(self.conv(self.relu(self.bn(inputs))))
