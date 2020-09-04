from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

from VGG_GAP import VGG_GAP

torch.cuda.set_device(0)

# Hyper parameters
epoch_num = 300
batch_size = 512
lr = 1e-4  # learning rate
img_channel = 3  # channel of generated image
workers = 2  # subprocess number for load the image
num_class = 10  # class number of your data
weight_decay = 1e-5

train_dataset_dir = './CIFAR10/'  # the path of your train_dataset
test_dataset_dir = './CIFAR10/'  # the path of your test_dataset

train_ds_size = 50000  # the size of your train dataset
test_ds_size = 10000  # the size of your test dataset

mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]


# data enhancement
data_transform = {'train':
                  transforms.Compose([
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize(mean, std)
                  ]),
                  'eval':
                  transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(mean, std)
                    ])}

# dataset
train_ds = torchvision.datasets.CIFAR10(train_dataset_dir, transform=data_transform['train'])

train_dl = DataLoader(train_ds, batch_size, True, num_workers=workers)

train_eval_ds = torchvision.datasets.CIFAR10(train_dataset_dir, transform=data_transform['eval'])

train_eval_dl = DataLoader(train_eval_ds, batch_size, num_workers=workers)


test_ds = torchvision.datasets.CIFAR10(test_dataset_dir, train=False, transform=data_transform['eval'])

test_dl = DataLoader(test_ds, batch_size, num_workers=workers)


# use cuda if you have GPU
net = nn.DataParallel(VGG_GAP([2, 2, 2, 2, 2], num_class, img_channel))

net = net.cuda()

# optimizer
opt = torch.optim.Adam(net.parameters(), lr=lr)  # optimizer for network

# loss function
loss_func = nn.CrossEntropyLoss()

# train the network
start = time()

for epoch in range(epoch_num):

    for step, (data, target) in enumerate(train_dl, 1):

        data, target = Variable(data).cuda(), Variable(target).cuda()

        outputs = net(data)

        loss = loss_func(outputs, target)

        opt.zero_grad()

        loss.backward()

        opt.step()

        if step % 20 == 0:

            net.eval()  # 使用过BN或者其他的话一定要记得在测试的时候使用这句话！！！！！

            test_acc, train_acc = 0, 0

            for test_step, (data, target) in enumerate(train_eval_dl, 1):

                data, target = Variable(data).cuda(), Variable(target).cuda()

                outputs = net(data)

                train_acc += sum(torch.max(outputs, 1)[1].data.cpu().numpy() == target.data.cpu().numpy())

            for test_step, (data, target) in enumerate(test_dl, 1):

                data, target = Variable(data).cuda(), Variable(target).cuda()

                outputs = net(data)

                test_acc += sum(torch.max(outputs, 1)[1].data.cpu().numpy() == target.data.cpu().numpy())

            train_acc /= train_ds_size
            test_acc /= test_ds_size

            net.train()  # 使用过BN或者其他的话一定要记得在测试的时候使用这句话！！！！！

            print('epoch:{}, step:{}, train_acc:{:.3f} %, test_acc:{:.3f} %, loss:{:.3f}, time:{:.1f} min'
                  .format(epoch, step, train_acc * 100, test_acc * 100, loss.item(), (time() - start) / 60))

torch.save(net.state_dict(), 'net{}-{}.pth'.format(epoch, step))  # 保存模型参数