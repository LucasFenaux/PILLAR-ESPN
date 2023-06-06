import torch.nn as nn


class ModulusNet_vgg16(nn.Module):
    def __init__(self, num_classes=10, pool='avg'):
        super().__init__()
        self.input_layer = nn.Identity()
        if pool == 'avg':
            self.pool = nn.AvgPool2d(2, 2)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.NumElemFlatten = 512

        self.fc1 = nn.Linear(self.NumElemFlatten, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.features = nn.Sequential(
            self.input_layer,
            self.conv1, self.relu, self.conv2, self.pool, self.relu,
            self.conv3, self.relu, self.conv4, self.pool, self.relu,
            self.conv5, self.relu, self.conv6, self.relu, self.conv7, self.pool, self.relu,
            self.conv8, self.relu, self.conv9, self.relu, self.conv10, self.pool, self.relu,
            self.conv11, self.relu, self.conv12, self.relu, self.conv13, self.pool, self.relu,
        )
        self.classifier = nn.Sequential(
            self.fc1, self.relu,
            self.fc2, self.relu,
            self.fc3
        )
        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4,
                            self.conv5, self.conv6, self.conv7, self.conv8,
                            self.conv9, self.conv10, self.conv11, self.conv12,
                            self.conv13]
        self.fc_layers = [self.fc1, self.fc2, self.fc3]
        self.linear_layers = self.conv_layers + self.fc_layers

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.NumElemFlatten)
        x = self.classifier(x)
        return x


class ModulusNet_vgg16_bn(nn.Module):
    def __init__(self, num_classes=10, pool="avg"):
        super().__init__()
        self.input_layer = nn.Identity()
        if pool == "avg":
            self.pool = nn.AvgPool2d(2, 2)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)

        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)

        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)

        self.NumElemFlatten = 512

        self.fc1 = nn.Linear(self.NumElemFlatten, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.features = nn.Sequential(
            self.input_layer,
            self.conv1, self.bn1, self.relu, self.conv2, self.bn2, self.pool, self.relu,
            self.conv3, self.bn3, self.relu, self.conv4, self.bn4, self.pool, self.relu,
            self.conv5, self.bn5, self.relu, self.conv6, self.bn6, self.relu, self.conv7, self.bn7, self.pool,
            self.relu,
            self.conv8, self.bn8, self.relu, self.conv9, self.bn9, self.relu, self.conv10, self.bn10, self.pool,
            self.relu,
            self.conv11, self.bn11, self.relu, self.conv12, self.bn12, self.relu, self.conv13, self.bn13, self.pool,
            self.relu,
        )
        self.classifier = nn.Sequential(
            self.fc1, self.relu,
            self.fc2, self.relu,
            self.fc3
        )
        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4,
                            self.conv5, self.conv6, self.conv7, self.conv8,
                            self.conv9, self.conv10, self.conv11, self.conv12,
                            self.conv13]
        self.fc_layers = [self.fc1, self.fc2, self.fc3]
        self.linear_layers = self.conv_layers + self.fc_layers

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.NumElemFlatten)
        x = self.classifier(x)
        return x
