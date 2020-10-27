import torch
import torch.nn as nn
import torch.nn.functional as F
import collections as col
import os
from os.path import join as pjoin

from .layers_torch import AllViewsConvLayer, AllViewsPad, AllViewsMaxPool, AllViewsAvgPool, AllViewsGaussianNoise


class BaselineBreastModel(nn.Module):

    def __init__(self, device, nodropout_probability=None, gaussian_noise_std=None, pretrained=False):
        super(BaselineBreastModel, self).__init__()
        self.conv_layer_dict = col.OrderedDict()

        # first conv sequence
        self.conv_layer_dict["conv1"] = AllViewsConvLayer(1, number_of_filters=32, filter_size=(3, 3), stride=(2, 2))

        # second conv sequence
        self.conv_layer_dict["conv2a"] = AllViewsConvLayer(32, number_of_filters=64, filter_size=(3, 3), stride=(2, 2))
        self.conv_layer_dict["conv2b"] = AllViewsConvLayer(64, number_of_filters=64, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv2c"] = AllViewsConvLayer(64, number_of_filters=64, filter_size=(3, 3), stride=(1, 1))

        # third conv sequence
        self.conv_layer_dict["conv3a"] = AllViewsConvLayer(64, number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv3b"] = AllViewsConvLayer(128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv3c"] = AllViewsConvLayer(128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1))

        # fourth conv sequence
        self.conv_layer_dict["conv4a"] = AllViewsConvLayer(128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv4b"] = AllViewsConvLayer(128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv4c"] = AllViewsConvLayer(128, number_of_filters=128, filter_size=(3, 3), stride=(1, 1))

        # fifth conv sequence
        self.conv_layer_dict["conv5a"] = AllViewsConvLayer(128, number_of_filters=256, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv5b"] = AllViewsConvLayer(256, number_of_filters=256, filter_size=(3, 3), stride=(1, 1))
        self.conv_layer_dict["conv5c"] = AllViewsConvLayer(256, number_of_filters=256, filter_size=(3, 3), stride=(1, 1))
        self._conv_layer_ls = nn.ModuleList(self.conv_layer_dict.values())

        # Pool, flatten, and fully connected layers
        self.all_views_pad = AllViewsPad()
        self.all_views_max_pool = AllViewsMaxPool()
        self.all_views_avg_pool = AllViewsAvgPool()

        self.fc1 = nn.Linear(256 * 4, 256 * 4)
        self.fc2 = nn.Linear(256 * 4, 4)

        self.gaussian_noise_layer = AllViewsGaussianNoise(gaussian_noise_std, device=device)
        self.dropout = nn.Dropout(p=1 - nodropout_probability)

        # NOTE: Added weight loading logic
        if pretrained:
            self.load_state_dict(torch.load('./pretrained_weights/gerasnet.pth'))

    def forward(self, x):
        # x = {
        #    'L-CC': array([....]),,
        #    'R-CC': array([....]),
        #    'L-MLO': array([....]),
        #    'R-MLO': array([....]),
        # }

        x = self.gaussian_noise_layer(x)

        # first conv sequence
        x = self.conv_layer_dict["conv1"](x)

        # second conv sequence
        x = self.all_views_max_pool(x, stride=(3, 3))
        x = self.conv_layer_dict["conv2a"](x)
        x = self.conv_layer_dict["conv2b"](x)
        x = self.conv_layer_dict["conv2c"](x)

        # third conv sequence
        x = self.all_views_max_pool(x, stride=(2, 2))
        x = self.conv_layer_dict["conv3a"](x)
        x = self.conv_layer_dict["conv3b"](x)
        x = self.conv_layer_dict["conv3c"](x)

        # WARNING: This is technically correct, but not robust to model architecture changes.
        x = self.all_views_pad(x, pad=(0, 1, 0, 0))

        # fourth conv sequence
        x = self.all_views_max_pool(x, stride=(2, 2))
        x = self.conv_layer_dict["conv4a"](x)
        x = self.conv_layer_dict["conv4b"](x)
        x = self.conv_layer_dict["conv4c"](x)

        # fifth conv sequence
        x = self.all_views_max_pool(x, stride=(2, 2))
        x = self.conv_layer_dict["conv5a"](x)
        x = self.conv_layer_dict["conv5b"](x)
        x = self.conv_layer_dict["conv5c"](x)
        x = self.all_views_avg_pool(x)

        # Pool, flatten, and fully connected layers
        x = torch.cat([
            x["L-CC"],
            x["R-CC"],
            x["L-MLO"],
            x["R-MLO"],
        ], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # NOTE: The last softmax layer was removed

        return x
