import torch.nn as nn
import torch.nn.functional as F


class ModulusNet_vgg16(nn.Module):
    def __init__(self, store_configs: dict, num_class=100):
        super().__init__()
        self.store_configs = store_configs

        self.input_layer = nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.conv7 = nn.Conv2d(256, 256, 3)

        self.conv8 = nn.Conv2d(256, 512, 3)
        self.conv9 = nn.Conv2d(512, 512, 3)
        self.conv10 = nn.Conv2d(512, 512, 3)

        self.conv11 = nn.Conv2d(512, 512, 3)
        self.conv12 = nn.Conv2d(512, 512, 3)
        self.conv13 = nn.Conv2d(512, 512, 3)
        self.NumElemFlatten = 512

        self.fc1 = nn.Linear(self.NumElemFlatten, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_class)

        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4,
                            self.conv5, self.conv6, self.conv7, self.conv8,
                            self.conv9, self.conv10, self.conv11, self.conv12,
                            self.conv13]
        self.fc_layers = [self.fc1, self.fc2, self.fc3]
        self.linear_layers = self.conv_layers + self.fc_layers

        for layer in self.linear_layers:
            layer.set_store_configs(self.store_configs)

    def forward_without_bn(self, x):
        x = self.input_layer(x)
        x = self.pool(self.conv2(self.conv1(x)))
        x = self.pool(self.conv4(self.conv3(x)))

        x = self.pool(self.conv7(self.conv6(F.relu(self.conv5(x)))))
        x = self.pool(self.conv10(self.conv9(F.relu(self.conv8(x)))))
        x = self.pool(self.conv13(self.conv12(F.relu(self.conv11(x)))))

        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        return self.forward_without_bn(x)
