import torch
import torch.nn as nn

import math

from Tricks.shakedrop import ShakeDrop

class Net(nn.Module):

    # Basic Building Block of PyramidNet (bottleneck version)
    class BottleneckBlock(nn.Module):

        def __init__(self, in_channel, out_channel, stride, p_shakedrop=1.0, downsample=None):
            super(Net.BottleneckBlock, self).__init__()
            
            self.shake_drop = ShakeDrop(p_shakedrop)
        
            # BN -> 1x1 conv -> BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 1x1 conv -> BN
            self.layers = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel * 4)
            )
            self.downsample = downsample

        def forward(self, x):
            
            out = self.layers(x)
            out = self.shake_drop(out)

            if self.downsample == None:
                shortcut = x
                shortcut_size = out.size()[2:4]
            else:
                shortcut = self.downsample(x) 
                shortcut_size = shortcut.size()[2:4]

            assert shortcut_size == out.size()[2:4]

            batch_size = out.size()[0]
            padding_channels = out.size()[1] - shortcut.size()[1]
            assert padding_channels >= 0

            if padding_channels > 0:
                padding = torch.zeros(
                    batch_size, 
                    padding_channels,  
                    shortcut_size[0], 
                    shortcut_size[1],
                    device=shortcut.device,
                    dtype=shortcut.dtype,
                    requires_grad=False
                )
                shortcut = torch.cat([shortcut, padding], dim=1)

            out = out + shortcut

            return out


    def __init__(self, depth, alpha):
        super(Net, self).__init__()

        self.temp_layer = nn.AvgPool2d((4, 4))

        assert ((depth - 2) % 9) == 0
        # each bottleneck block contain 3 conv layers
        
        self.n = int((depth - 2) / 9) # the number of building block of each conv group 2,3,4
        self.additive_rate = alpha / (3.0 * self.n) # additive rate of feature map dimension
        # print(self.additive_rate)

        self.ps_shakedrop = [1 - (1.0 - (0.5 / (3 * self.n)) * (i + 1)) for i in range(3 * self.n)]

        # Conv Group 1
        # self.conv_group_1 = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(16)
        # )
        self.conv_group_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # out 32 x 32 x 16
        self.in_channels = 16
        self.featuremap_dim = 16

        # Conv Group 2,3,4
        self.conv_group_2 = self.conv_group_maker(2)
        self.conv_group_3 = self.conv_group_maker(3)
        self.conv_group_4 = self.conv_group_maker(4)

        # Final Group
        self.final_group = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.AvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(self.in_channels, 10)
        )

        # He Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n_para = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n_para))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        # Linear layer would initialize with Kaiming method automatically


    def conv_group_maker(self, id):

        if id == 2:
            stride = 1
            downsample = None
        else: # Conv Group 3/4
            stride = 2
            downsample = nn.AvgPool2d((2,2), stride=(2,2), ceil_mode=True)

        layers = []

        temp = self.featuremap_dim + self.additive_rate
        layers.append(self.BottleneckBlock(self.in_channels, int(round(temp)), stride, self.ps_shakedrop[0], downsample))
        self.featuremap_dim = temp
        self.in_channels = int(round(temp)) * 4
        for _ in range(1, self.n):
            temp = self.featuremap_dim + self.additive_rate
            layers.append(self.BottleneckBlock(self.in_channels, int(round(temp)), 1, self.ps_shakedrop[_]))
            self.featuremap_dim = temp
            self.in_channels = int(round(temp)) * 4

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv_group_1(x)
        out = self.conv_group_2(out)
        out = self.conv_group_3(out)
        out = self.conv_group_4(out)
        
        out = self.final_group(out)

        return out