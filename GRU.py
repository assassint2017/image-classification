"""train on MNIST"""

import torch.nn as nn


class GRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn1 = nn.GRU(28, 32, batch_first=True)

        self.rnn2 = nn.GRU(32, 64, batch_first=True)

        self.rnn3 = nn.GRU(64, 128, batch_first=True)

        self.clf = nn.Linear(128, 10)

    def forward(self, inputs):

        inputs = inputs.view(-1, 28, 28)

        outputs, h_n = self.rnn1(inputs)
        outputs, h_n = self.rnn2(outputs)
        outputs, h_n = self.rnn3(outputs)

        outputs = self.clf(outputs[:, -1, :])

        return outputs
