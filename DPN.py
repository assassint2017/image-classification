#_*_ coding:utf-8 _*_

import torch
import torch.nn as nn


# 鉴于定义一个DPN网络的参数比较多，所以特地写一个函数方便定义网络结构
def dpn92(color_channels, num_class):

    return DPN(color_channels, 64, 96, [3, 4, 20, 3], 32, [16, 32, 24, 128], num_class)


def dpn98(color_channels, num_class):

    return DPN(color_channels, 96, 160, [3, 6, 20, 3], 40, [16, 32, 32, 128], num_class)


def dpn131(color_channels, num_class):

    return DPN(color_channels, 128, 160, [4, 8, 28, 3], 40, [16, 32, 32, 128], num_class)


def dpn107(color_channels, num_class):

    return DPN(color_channels, 128, 200, [4, 8, 20, 3], 50, [20, 64, 64, 128], num_class)


class DPN(nn.Module):
    """双通道网路

    网络起始的时候也是一个卷积和池化来降维
    然后跟着4个stage，最后用GAP和FC来做分类
    在主干通道中，四种网络每一个stage输出的通道数量都是：256 512 1024 2048
    代码中共包含4种不同的网络:

    DPN-92: init_feature=64, init_conv_feature=96, block_nums=(3,4,20,3), groups=32, growth_rate=(16,32,24,128)
    DPN-98: init_feature=96, init_conv_feature=160, block_nums=(3,6,20,3), groups=40, growth_rate=(16,32,32,128)
    DPN-131: init_feature=128, init_conv_feature=160, block_nums=(4,8,28,3), groups=40, growth_rate=(16,32,32,128)
    DPN-107: init_feature=128, init_conv_feature=200, block_nums=(4,8,20,3), groups=50,  growth_rate=(20,64,64,128)
    """
    def __init__(self, color_channels, init_feature, init_conv_feature, block_nums, groups, growth_rate, class_nums):
        """参数说明!

        :param color_channels: 原始图像数据中的通道数量
        :param init_feature: 第一个卷积层的通道数量
        :param init_conv_feature: 第一个stage中的第一个卷积层的通道数量
        :param block_nums: 每一个stage中包含的block的数量
        :param groups: ResNext主干通道中分组卷积的分组数量
        :param growth_rate: DenseNet分支通道中卷积层的增长率
        :param class_nums: 一共要分成多少个类别
        """
        super(DPN, self).__init__()

        self.init_conv_feature = init_conv_feature  # 每一个stage中间卷积层的通道数量

        self.init_feature = init_feature  # 起始卷积层的通道数量

        self.groups = groups  # 分组卷积的分组数量

        self.inchannels = 0  # 记录每个输入层的卷积通道数量

        self.conv1 = nn.Sequential(  # 第一层的结构需要根据实际问题来调整
            nn.Conv2d(color_channels, init_feature, 3, 1, 1, bias=False),
            nn.BatchNorm2d(init_feature),
            nn.ReLU()

            #nn.MaxPool2d(3, 2, 1)
        )

        self.stage1 = self.make_stage(1, block_nums[0], growth_rate[0])

        self.stage2 = self.make_stage(2, block_nums[1], growth_rate[1], growth_rate[0])

        self.stage3 = self.make_stage(3, block_nums[2], growth_rate[2], growth_rate[1])

        self.stage4 = self.make_stage(4, block_nums[3], growth_rate[3], growth_rate[2])

        self.final_bn = nn.Sequential(
            nn.BatchNorm2d(2048 + (block_nums[3] + 2) * growth_rate[3]),
            nn.ReLU()
        )

        self.clf = nn.Sequential(
            nn.Linear(2048 + (block_nums[3] + 2) * growth_rate[3], class_nums),
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

    def make_stage(self, stage, block_num, growth_rate, last_growth_rate=0):
        """

        :param stage: 第几个stage
        :param block_num: stage中有多少个block
        :param growth_rate: 本层stage中的增长率
        :param last_growth_rate: 上一层stage中的增长率
        :return: 代表一个stage的Sequential
        """

        layer = []

        for i in range(block_num):

            if i == 0 and stage == 1:

                stride = 1
                channel_match = True
                self.inchannels = self.init_feature

            elif i == 0:

                stride = 2
                channel_match = True
                self.inchannels += last_growth_rate

            else:

                stride = 1
                channel_match = False
                self.inchannels = 2 ** (8 + (stage - 1)) + (3 + (i - 1)) * growth_rate

            midchannles = self.init_conv_feature * (2 ** (stage - 1))

            outchannels = 2 ** (8 + (stage - 1))

            layer += [Dualpath_block(self.inchannels, midchannles, outchannels, stride, self.groups, growth_rate, channel_match)]

        return nn.Sequential(*layer)

    def forward(self, inputs):

        outputs = self.conv1(inputs)

        outputs = self.stage1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.stage4(outputs)

        outputs = torch.cat(outputs, dim=1)
        outputs = self.final_bn(outputs)

        outputs = nn.functional.avg_pool2d(outputs, 4)

        outputs = outputs.view(outputs.size(0), -1)

        return self.clf(outputs)


class Dualpath_block(nn.Module):

    def __init__(self, inchannels, midchannels, outchannels, stride, gruops, growth_rate, channel_match=False):
        """参数说明

        :param inchannels: block的输入卷积通道数量
        :param midchannels: block中间层的卷积通道数量
        :param outchannels: block主干通道中卷积输出层的通道数量
        :param stride: 中间层的步长
        :param gruops: 分组卷积的分组数量
        :param growth_rate: 增长率
        :param channel_match: 是否需要1*1的卷积来匹配输入输出的维度，默认为不需要
        """
        super(Dualpath_block, self).__init__()

        self.outchannels = outchannels

        self.channel_math = channel_match

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(inchannels),
            nn.ReLU(),
            nn.Conv2d(inchannels, midchannels, 1, bias=False)
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(midchannels),
            nn.ReLU(),
            nn.Conv2d(midchannels, midchannels, 3, stride, 1, groups=gruops, bias=False)
        )

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(midchannels),
            nn.ReLU(),
            nn.Conv2d(midchannels, outchannels + growth_rate, 1, bias=False)
        )

        if channel_match:  # 对于分支通道，保留两倍增长率的卷积通道数量

            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(inchannels),
                nn.ReLU(),
                nn.Conv2d(inchannels, outchannels + 2 * growth_rate, 1, stride, bias=False)
            )

    def forward(self, inputs):

        if isinstance(inputs, tuple):

            summ_inputs = inputs[0]
            dense_inputs = inputs[1]

            # 将两个通道合并在一起
            inputs = torch.cat((summ_inputs, dense_inputs), dim=1)

        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)

        # 将两个通道分开
        summ = outputs[:, :self.outchannels, :, :]
        dense = outputs[:, self.outchannels:, :, :]

        if self.channel_math:

            proj = self.shortcut(inputs)

            summ_inputs = proj[:, :self.outchannels, :, :]
            dense_inputs = proj[:, self.outchannels:, :, :]

        summ += summ_inputs
        dense = torch.cat((dense, dense_inputs), dim=1)

        return summ, dense
