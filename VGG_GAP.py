#_*_ coding:utf-8 _*_

import torch.nn as nn


class VGG_GAP(nn.Module):  # 根据VGG的设计原则来进行改编
    """attention!

    和原始的VGG相比，去掉了最后的全连接层,改成了全局平均池化，网络一共分为5个stage
    每一个stage的通道数量分别为:64,128,256,512,512。所以区别仅在于每个stage使用的卷积层的数量
    下面所列出不同深度的VGG网络每个stage所包含的卷积层的数量

    VGG11:1,1,2,2,2
    VGG13:2,2,2,2,2
    VGG16:2,2,3,3,3
    VGG19:2,2,4,4,4
    """
    def __init__(self, blocks, num_class, color_channels):
        """参数说明

        :param blocks: 每一个stage里所包含的block的数量
        :param num_class: 一共有多少类别
        :param color_channels: 图像数据有多少通道
        """

        super(VGG_GAP, self).__init__()

        self.color_channels = color_channels  # 每一个stage之后是够加入池化需要根据实际图像尺寸来调整！

        self.stage1 = nn.Sequential(
            self.make_stage(blocks[0], 1),
            nn.MaxPool2d(3, 2, 1)
        )

        self.stage2 = nn.Sequential(
            self.make_stage(blocks[1], 2),
            nn.MaxPool2d(3, 2, 1)
        )

        self.stage3 = nn.Sequential(
            self.make_stage(blocks[2], 3),
            nn.MaxPool2d(3, 2, 1)
        )

        self.stage4 = nn.Sequential(
            self.make_stage(blocks[3], 4)
            #nn.MaxPool2d(3, 2, 1)
        )

        self.stage5 = nn.Sequential(
            self.make_stage(blocks[4], 5)
            #nn.MaxPool2d(3, 2, 1)
        )

        self.clf = nn.Sequential(
            nn.Linear(512, num_class),
            nn.Softmax(dim=1)
        )

        # 做参数初始化

        for layer in self.modules():

            if isinstance(layer, nn.Conv2d):

                nn.init.kaiming_normal_(layer.weight.data)

            elif isinstance(layer, nn.BatchNorm2d):

                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

            elif isinstance(layer, nn.Linear):

                layer.bias.data.zero_()

    def forward(self, inputs):

        outputs = self.stage1(inputs)

        outputs = self.stage2(outputs)

        outputs = self.stage3(outputs)

        outputs = self.stage4(outputs)

        outputs = self.stage5(outputs)

        outputs = nn.functional.avg_pool2d(outputs, 4)  # 需要根据实际图像尺寸来调整！
        outputs = outputs.view(outputs.size(0), -1)

        return self.clf(outputs)

    def make_stage(self, block_num, stage):

        outchannels = 64 * (2 ** (stage - 1)) if stage < 5 else 512

        inchannels = None

        if stage == 1:

            inchannels = self.color_channels

        else:

            stage -= 1
            inchannels = 64 * (2 ** (stage - 1))

        layer = []

        for i in range(block_num):

            if i == 0:

                layer += [Basic_block(inchannels, outchannels)]

            else:

                layer += [Basic_block(outchannels, outchannels)]

        return nn.Sequential(*layer)


class Basic_block(nn.Module):

    def __init__(self, inchannels, outchannels):
        super(Basic_block, self).__init__()

        self.conv = nn.Conv2d(inchannels, outchannels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU()

    def forward(self, inputs):

        return self.relu(self.bn(self.conv(inputs)))
