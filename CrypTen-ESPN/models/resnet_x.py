'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['MiniONN', 'ConvNet', 'ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


class ConvNet(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MiniONN(nn.Module):
    def __init__(self, num_classes=10, to_quant: bool = False, init_weights=True, use_batch_norm=True):
        super().__init__()

        self.to_quant = to_quant
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1))
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.bn7 = nn.BatchNorm2d(16)

        self.fully_connected_layer = nn.Linear(in_features=1024, out_features=num_classes)

        self.mean_pooling = nn.AvgPool2d(kernel_size=(2, 2))
        self.use_batch_norm = use_batch_norm
        self.relu = nn.ReLU()
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.to_quant:
            x = self.quant(x)
        if self.use_batch_norm:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.mean_pooling(self.relu(self.bn2(self.conv2(x))))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.mean_pooling(self.relu(self.bn4(self.conv4(x))))
            x = self.relu(self.bn5(self.conv5(x)))
            x = self.relu(self.bn6(self.conv6(x)))
            x = self.relu(self.bn7(self.conv7(x)))
        else:
            x = self.relu(self.conv1(x))
            x = self.mean_pooling(self.relu(self.conv2(x)))
            x = self.relu(self.conv3(x))
            x = self.mean_pooling(self.relu(self.conv4(x)))
            x = self.relu(self.conv5(x))
            x = self.relu(self.conv6(x))
            x = self.relu(self.conv7(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fully_connected_layer(x)

        if self.to_quant:
            x = self.dequant(x)

        return x


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_weights=True):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.quantized = False
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        if init_weights:
            self.apply(_weights_init)

    def quantize(self):
        self.quantized = True

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.quantized:
            x = self.quant(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.quantized:
            out = self.dequant(out)

        return out


def resnet20(num_classes, init_weights=True):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, init_weights=init_weights)


def resnet32(num_classes, init_weights=True):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, init_weights=init_weights)


def resnet44(num_classes, init_weights=True):
    return ResNet(BasicBlock, [7, 7, 7], num_classes, init_weights=init_weights)


def resnet56(num_classes, init_weights=True):
    return ResNet(BasicBlock, [9, 9, 9], num_classes, init_weights=init_weights)


def resnet110(num_classes, init_weights=True):
    return ResNet(BasicBlock, [18, 18, 18], num_classes, init_weights=init_weights)


def resnet1202(num_classes, init_weights=True):
    return ResNet(BasicBlock, [200, 200, 200], num_classes, init_weights=init_weights)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
