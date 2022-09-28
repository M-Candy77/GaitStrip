import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from torch.nn.parameter import Parameter

import torch.nn as nn 
#E3D模块
class E3D(nn.Module):
    def __init__(self, inplanes, planes, bias=False, **kwargs):
        super(E3D, self).__init__()
        self.conv3d1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), bias=bias, padding=(1, 1, 1))
        self.conv3d2 = nn.Conv3d(inplanes, planes, kernel_size=(1, 3, 3), bias=bias, padding=(0, 1, 1))
        self.conv3d3 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 3), bias=bias, padding=(1, 0, 1))
        self.conv3d4 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 1), bias=bias, padding=(1, 1, 0))

    def forward(self, x):
        out1 = self.conv3d1(x)
        out2 = self.conv3d2(x)
        out3 = self.conv3d3(x)
        out4 = self.conv3d4(x)

        out = out1 + out2 + out3 +out4

        out = F.leaky_relu(out, inplace=True)


        return out

def gem(x, p=6.5, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (1, x.size(-1))).pow(1./p)

class GeM(nn.Module):

    def __init__(self, p=6.5, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class BasicConv3d3(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(BasicConv3d3, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), bias=bias, dilation=(dilation, 1, 1), padding=(dilation, 1, 1))

    def forward(self, x):
        out1 = self.conv1(x)
        out = F.leaky_relu(out1, inplace=True)
        return out

class LocaltemporalAG(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(LocaltemporalAG, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), stride=(3,1,1), bias=bias,padding=(0, 0, 0))

    def forward(self, x):
        out1 = self.conv1(x)
        out = F.leaky_relu(out1, inplace=True)
        return out

class C3D_VGG(nn.Module):

    def __init__(self, num_classes=74):
        super(C3D_VGG, self).__init__()

        _set_channels = [32, 64, 128, 128] 

        # --------------------------------  2d gei---------------------------------------
        self.conv2dlayer1a = BasicConv3d3(3, _set_channels[0])
        self.pool1 = LocaltemporalAG(_set_channels[0], _set_channels[0])

        self.conv2dlayer2a = E3D(_set_channels[0], _set_channels[1])
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2dlayer3a = E3D(_set_channels[1], _set_channels[2])
        self.conv2dlayer3b = E3D(_set_channels[2], _set_channels[2])

        self.conv2dlayer4a = E3D(_set_channels[1], _set_channels[2])
        self.conv2dlayer4b = E3D(_set_channels[2], _set_channels[2])

        self.Gem1 = GeM()
        self.Gem2 = GeM()

        self.bin_num32 = [32]
        self.bin_num64 = [64]
        
        
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros((sum(self.bin_num32)), _set_channels[2], _set_channels[2]))),
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros((sum(self.bin_num64)), _set_channels[2], _set_channels[2]))),
                    ])

        self.bn2 = nn.BatchNorm1d(_set_channels[2])

        self.fc2 = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros((sum(self.bin_num32+self.bin_num64)), _set_channels[2], num_classes)))])


        self.relu = nn.ReLU()
        for m in self.modules():
            # print('---')
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        # print(x.shape)
        n, c, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 1, 2, 1, 1)

        # ----------------2d--------------------
        x2dl1a = self.conv2dlayer1a(x)
        # x2dl1b = self.conv2dlayer1b(x2dl1a)
        x2dp1 = self.pool1(x2dl1a)
        # print('x2dp1-',x2dp1.shape)

        x2dl2a = self.conv2dlayer2a(x2dp1)
        x2dp2 = self.pool2(x2dl2a)
        x2dl3a = self.conv2dlayer3a(x2dp2)
        x2dl3b = self.conv2dlayer3b(x2dl3a)
        x2dl4a = self.conv2dlayer4a(x2dl2a)
        x2dl4b = self.conv2dlayer4b(x2dl4a)


        x2dl3b = torch.max(x2dl3b, 2)[0]
        x2dl4b = torch.max(x2dl4b, 2)[0]


        _, c2dl3, _, _ = x2dl3b.size()
        _, c2dl4, _, _ = x2dl4b.size()


        feature1 = list()
        for num_bin in self.bin_num32:

            z = x2dl3b.view(n, c2dl3, num_bin, -1).contiguous()
            z = self.Gem1(z).squeeze(-1)
            feature1.append(z)
        feature1 = torch.cat(feature1, 2).permute(2, 0, 1).contiguous()
        feature1 = feature1.matmul(self.fc_bin[0])
        feature1 = feature1.permute(1, 2, 0).contiguous()

        feature2 = list()
        for num_bin in self.bin_num64:
            z = x2dl4b.view(n, c2dl4, num_bin, -1).contiguous()
            z = self.Gem2(z).squeeze(-1)
            feature2.append(z)
        feature2 = torch.cat(feature2, 2).permute(2, 0, 1).contiguous()
        feature2 = feature2.matmul(self.fc_bin[1])
        feature2 = feature2.permute(1, 2, 0).contiguous()

        feature = torch.cat((feature1,feature2),dim=2)

        featurebn2 = self.bn2(feature)

        feature = featurebn2.permute(2, 0, 1).contiguous()
        feature = feature.matmul(self.fc2[0])
        feature = feature.permute(1, 0, 2).contiguous()


        return featurebn2,feature



def params_count(net):
    list1 = []
    for p in net.parameters():
        list1.append(p)
    n_parameters = sum(p.numel() for p in net.parameters())
    print('-----Model param: {:.5f}M'.format(n_parameters / 1e6))
    return n_parameters



def c3d_vgg_Fusion(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = C3D_VGG(**kwargs)
    return model


if __name__ == "__main__":
    net = c3d_vgg_Fusion(num_classes=74)
    print(params_count(net))
    with torch.no_grad():
        x = torch.ones(4 * 3 * 32 * 64 * 44).reshape(4, 3, 32, 64, 44)
        print('x=', x.shape)
        a,b = net(x)
        print('a,b=',a.shape,b.shape)
        