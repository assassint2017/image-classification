#_*_ coding:utf-8 _*_

import torch.nn as nn


class ResNet(nn.Module):
    """attention!

    需要根据实际的图像尺寸大小进行调整
    这个版本的ResNet是缩小了 8 倍！
    网络开头有一个卷积层，之后跟着4个stage,每一个stage的输出通道数量都可以用下式表达
    64,128,256,512 * expansion。对于basic来说扩展系数为1，对于bottleneck来说系数为4
    下面列举出来的是不同深度网络每一个stage所使用的block数量

    ResNet-18:   2,2,2,2
    ResNet-34:   3,4,6,3

    ResNet-50:   3,4,6,3
    ResNet-101:  3,4,23,3
    ResNet-152:  3,8,36,3
    """

    def __init__(self, init_feature, init_conv_feature, block_type, block_nums, color_channels, num_class):
        """参数说明

        :param init_feature: 最初始的一个卷积层有多少个通道
        :param init_conv_feature: 第一个stage的第一个卷积层的通道数量
        :param block_type: block的类型，是原始类型，还是bottleneck
        :param block_nums: 每一个stage中包含的block的数量
        :param color_channels: 原始图像数据中的通道数量
        :param num_class: 一共要分成多少个类别
        """
        super(ResNet, self).__init__()

        self.init_conv_feature = init_conv_feature

        expansion = 4 if block_type == Bottleneck_block else 1

        self.conv1 = nn.Sequential(     # 第一层还是需要根据具体的任务来调整的
            nn.Conv2d(color_channels, init_feature, 3, 1, 1, bias=False),
            nn.BatchNorm2d(init_feature),
            nn.ReLU()

            #nn.MaxPool2d(3, 2, 1)
        )

        self.stage1 = self.make_stage(block_type, block_nums[0], 1)

        self.stage2 = self.make_stage(block_type, block_nums[1], 2)

        self.stage3 = self.make_stage(block_type, block_nums[2], 3)

        self.stage4 = nn.Sequential(
            self.make_stage(block_type, block_nums[3], 4),
            nn.AvgPool2d(4)  # 需要根据具体的任务来调整的
        )

        self.clf = nn.Sequential(
            nn.Linear(512 * expansion, num_class),
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

    def make_stage(self, block_type, block_nums, stage):

        outchannels = 64 * (2 ** (stage - 1))
        inputchannels, stride = None, 1

        layer = []

        for i in range(block_nums):

            if stage == 1 and i == 0:

                inputchannels = self.init_conv_feature
                channel_math = True

            elif i == 0:

                inputchannels = 64 * (2 ** (stage - 2)) if block_type == Basic_block else 64 * (2 ** (stage - 2)) * 4
                stride = 2
                channel_math = True

            else:

                inputchannels = 64 * (2 ** (stage - 1)) if block_type == Basic_block else 64 * (2 ** (stage - 1)) * 4
                stride = 1
                channel_math = False

            layer += [block_type(inputchannels, outchannels, stride, channel_math)]

        return nn.Sequential(*layer)


class Basic_block(nn.Module):  # 每一个block有两个卷积层

    def __init__(self, inchannels, outchannels, stride, channel_math):
        super(Basic_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(outchannels, outchannels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannels)
        )

        if channel_math == False:

            self.shortcut = nn.Sequential()

        else:

            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannels, outchannels, 1, stride, bias=False),
                nn.BatchNorm2d(outchannels)
            )

    def forward(self, inputs):

        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs += self.shortcut(inputs)

        return nn.functional.relu(outputs)


class Bottleneck_block(nn.Module):  # 每一个block有三个卷积层

    def __init__(self, inchannels, outchannles, stride, channel_math):
        super(Bottleneck_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannels, outchannles, 1, bias=False),
            nn.BatchNorm2d(outchannles),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(outchannles, outchannles, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannles),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(outchannles, outchannles * 4, 1, bias=False),
            nn.BatchNorm2d(outchannles * 4),
        )

        if channel_math == False:

            self.shortcut = nn.Sequential()

        else:

            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannels, outchannles * 4, 1, stride, bias=False),
                nn.BatchNorm2d(outchannles * 4)
            )

    def forward(self, inputs):

        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)

        outputs += self.shortcut(inputs)

        return nn.functional.relu(outputs)
