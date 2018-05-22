#_*_ coding:utf-8 _*_

import torch.nn as nn
import torch.nn.functional as F


class SEmodule(nn.Module):
    """attention!

    SENet与其说是一个网络，倒不如说是一个模块，一种注意力模型的思想

    """
    def __init__(self, inchannels, reduction_ratio):
        """参数说明

        :param inchannels: 输入的特征图的通道的个数
        :param reduction_ratio: 控制两个全连接层之间的神经元个数
        """
        super(SEmodule, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(inchannels, inchannels // reduction_ratio),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(inchannels // reduction_ratio, inchannels),
            nn.Sigmoid()
        )

    def forward(self, inputs):

        outputs = F.avg_pool2d(inputs, (inputs.size(2), inputs.size(3)))

        outputs = outputs.view(outputs.size(0), -1)

        outputs = self.fc1(outputs)

        outputs = self.fc2(outputs)

        outputs = outputs.view(outputs.size(0), outputs.size(1), 1, 1)

        return inputs * outputs


