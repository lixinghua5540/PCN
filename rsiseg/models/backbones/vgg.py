import os
import numpy as np
import time
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Vgg19(nn.Module):
    def __init__(self, input_channel=3,img_size=256,num_class=3,vgg19_npy_path=None):
        super(Vgg19, self).__init__()
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3,3], stride=1, padding=1),
            #nn.BatchNorm2d(64),  # default parameter：nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            nn.LeakyReLU(0.1,inplace=True)
        )

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1,inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1,inplace=True)
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1,inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,inplace=True)
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,inplace=True)
        )

        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,inplace=True)
        )

        self.conv3_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,inplace=True)
        )

        self.conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,inplace=True)
        )

        self.conv4_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,inplace=True)
        )
        self.conv4_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,inplace=True)
        )

        self.conv5_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,inplace=True)
        )

        self.conv5_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,inplace=True)
        )

        self.conv5_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.fc6 = nn.Sequential(
            nn.Linear(int(512 * img_size * img_size / 32 / 32), 4096),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(p=0.5)  # 默认就是0.5
        )

        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(p=0.5)
        )

        self.conv_list = [self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2, self.conv3_1, self.conv3_2, self.conv3_3,
                          self.conv3_4, self.conv4_1, self.conv4_2, self.conv4_3, self.conv4_4, self.conv5_1, self.conv5_2,
                          self.conv5_3, self.conv5_4]
        self.conv_name = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3",
                          "conv3_4", "conv4_1", "conv4_2", "conv4_3", "conv4_4", "conv5_1", "conv5_2",
                          "conv5_3", "conv5_4"]
        #self.fc_list = [self.fc6, self.fc7, self.fc8]
    def get_conv_filter(self, name):
        return torch.Tensor(self.data_dict[name][0])

    def get_bias(self, name):
        return torch.Tensor(self.data_dict[name][1])

    def load_dict(self):
        for i in range(len(self.conv_list)):
            cdata=self.conv_list[i]
            cname=self.conv_name[i]
            conv=self.get_conv_filter(cname)
            bias=self.get_bias(cname)
            convT=conv.permute(3,2,0,1)
            cdata[0].weight.data=convT#.to(self.device[0])
            cdata[0].bias.data=bias#.to(self.device[0])
        for param in self.parameters():
            param.requires_grad = False

        
    def forward(self, input):
        """Standard forward."""
        x=input
        Layer_out=[]
        for i in range(len(self.conv_list)):
            x = self.conv_list[i](x)
            if (i+2)%4==0 or (i==11)  or (i==0):
                Layer_out.append(x)

        return Layer_out[1], Layer_out[2], Layer_out[3]
