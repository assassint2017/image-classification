#_*_ coding:utf-8 _*_

import torch.nn as nn


class ResNeXt(nn.Module):
    """attention!

    本网络的参数相对较小的32*4d结构
    需要根据实际的图像尺寸大小进行调整
    这个版本的ResNeXt是缩小了 8 倍！

    网络开头有一个卷积层，之后跟着4个stage,每一个stage的输出通道数量都可以用下式表达
    256, 512, 1024, 2048
    下面列举出来的是不同深度网络每一个stage所使用的block数量

    ResNeXt-50:   3,4,6,3
    ResNeXt-101:  3,4,23,3
    ResNeXt-152:  3,8,36,3
    """

    def __init__(self, init_feature, init_conv_deature, block_nums, color_channels, num_class, groups=32):
        """参数说明

        :param init_feature: 最初始的一个卷积层有多少个通道
        :param init_conv_deature: 第一个stage中卷积层的通道数量
        :param block_nums: 每一个stage中包含的block的数量
        :param color_channels: 原始图像数据中的通道数量
        :param num_class: 一共要分成多少个类别
        :param groups: 分组卷积的数量,默认为32
        """
        super(ResNeXt, self).__init__()

        self.init_conv_feature = init_conv_deature

        self.conv1 = nn.Sequential(     # 第一层还是需要根据具体的任务来调整的
            nn.Conv2d(color_channels, init_feature, 3, 1, 1, bias=False),
            nn.BatchNorm2d(init_feature),
            nn.ReLU()

            #nn.MaxPool2d(3, 2, 1)
        )

        self.stage1 = self.make_stage(block_nums[0], 1, groups)

        self.stage2 = self.make_stage(block_nums[1], 2, groups)

        self.stage3 = self.make_stage(block_nums[2], 3, groups)

        self.stage4 = nn.Sequential(
            self.make_stage(block_nums[3], 4, groups),
            nn.AvgPool2d(4)  # 需要根据具体的任务来调整的
        )

        self.clf = nn.Sequential(
            nn.Linear(2048, num_class),
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

        outputs = self.stage1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.stage4(outputs)

        outputs = outputs.view(outputs.size(0), -1)

        return self.clf(outputs)

    def make_stage(self, block_nums, stage, groups=32):

        outchannels = 64 * (2 ** stage)
        inputchannels, stride = None, 1

        layer = []

        for i in range(block_nums):

            if stage == 1 and i == 0:

                inputchannels = self.init_conv_feature
                channel_math = True

            elif i == 0:

                inputchannels = 64 * (2 ** (stage - 2)) * 4
                stride = 2
                channel_math = True

            else:

                inputchannels = 64 * (2 ** (stage - 1)) * 4
                stride = 1
                channel_math = False

            layer += [Bottleneck_block(inputchannels, outchannels, stride, groups, channel_math)]

        return nn.Sequential(*layer)


class Bottleneck_block(nn.Module):  # 每一个block有三个卷积层

    def __init__(self, inchannels, outchannles, stride, groups, channel_math):
        super(Bottleneck_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannels, outchannles, 1, bias=False),
            nn.BatchNorm2d(outchannles),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(outchannles, outchannles, 3, stride, 1, bias=False, groups=groups),
            nn.BatchNorm2d(outchannles),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(outchannles, outchannles * 2, 1, bias=False),
            nn.BatchNorm2d(outchannles * 2),
        )

        if channel_math is False:

            self.shortcut = nn.Sequential()

        else:

            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannels, outchannles * 2, 1, stride, bias=False),
                nn.BatchNorm2d(outchannles * 2)
            )

    def forward(self, inputs):

        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)

        outputs += self.shortcut(inputs)

        return nn.functional.relu(outputs)
