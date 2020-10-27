import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.models as models
from models.gerasnet import BaselineBreastModel as GerasNet


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# class MyModel(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         self.basemodel_cc = GerasNet(config['device'], nodropout_probability=1.0, gaussian_noise_std=0.0, pretrained=True)
#         # freeze all but the last layer
#         for param in self.basemodel_cc.parameters():
#             param.requires_grad = False
#         self.basemodel_cc.fc2 = nn.Linear(256 * 4, 1)
#         self.float()

#     def forward(self, x):
#         lcc, rcc, lmlo, rmlo = x.transpose(0, 1)
#         x = {
#             "L-CC": lcc,
#             "R-CC": rcc,
#             "L-MLO": lmlo,
#             "R-MLO": rmlo,
#         }
#         x = self.basemodel_cc(x)
#         return x

# class MyModel(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         # resnet18
#         # self.basemodel_cc = ResNet(BasicBlock, [2, 2, 2, 2])
#         # self.basemodel_cc.fc = nn.Identity()

#         # densenet121 ---------------------

#         # self.basemodel_cc = models.densenet121(pretrained=True)
#         # tmp = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # sd = self.basemodel_cc.features.conv0.state_dict()
#         # sd['weight'] = sd['weight'][:, [0], :, :]
#         # tmp.load_state_dict(sd)

#         # self.basemodel_cc.features.conv0 = tmp
#         # self.basemodel_cc.classifier = nn.Identity()

#         # ---------------------------------

#         # ---------------------------------

#         # self.basemodel_mlo = ResNet(BasicBlock, [2, 2, 2, 2])
#         # self.basemodel_mlo.fc = nn.Identity()
#         # self.basemodel_mlo = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16))
#         # self.basemodel_mlo.classifier = nn.Identity()

#         self.classifier = nn.Linear(1024, config['n_classes'])
#         # self.classifier = nn.Linear(512, config['n_classes'])
#         # self.classifier = nn.Linear(2048, config['n_classes'])
#         self.float()

#     def forward(self, x):
#         lcc, rcc, lmlo, rmlo = x.transpose(0, 1)

#         lcc = self.basemodel_cc(lcc)
#         rcc = self.basemodel_cc(rcc)
#         lmlo = self.basemodel_cc(lmlo)
#         rmlo = self.basemodel_cc(rmlo)

#         both = (lcc + rcc + lmlo + rmlo) / 4

#         # lcc = self.basemodel_cc(lcc)
#         # rcc = self.basemodel_cc(rcc)
#         # cc = (lcc + rcc) / 2

#         # lmlo = self.basemodel_mlo(lmlo)
#         # rmlo = self.basemodel_mlo(rmlo)
#         # mlo = (lmlo + rmlo) / 2

#         # both = torch.cat([cc, mlo], 1)

#         x = self.classifier(both)

#         return x


class MyDenseNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.basemodel_cc = models.densenet121(pretrained=True)
        tmp = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        sd = self.basemodel_cc.features.conv0.state_dict()
        sd['weight'] = sd['weight'][:, [0, 1], :, :]
        tmp.load_state_dict(sd)

        self.basemodel_cc.features.conv0 = tmp
        self.basemodel_cc.classifier = nn.Identity()

        # self.classifier = nn.Linear(1024, 3)
        self.classifier = nn.Linear(1024, 1)

        self.float()

    def forward(self, xs):
        ys = []
        for x in xs:
            ys.append(self.basemodel_cc(x))
        all_x = torch.stack(ys)
        avg_x = torch.mean(all_x, dim=0)

        x = self.classifier(avg_x)

        return x


def get_model(config):
    # return nn.Sequential(MyDenseNet(config), nn.Softmax())
    return MyDenseNet(config)
