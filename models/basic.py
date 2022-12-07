import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class reorg_layer(nn.Module):  ##空间重排操作用于将特征图的大小减半，同时将其通道数增加四倍。
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride  ##划分成了stride * stride个小块
        
        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        ##_height * _width表示将原始特征图划分出来的小块数目, self.stride * self.stride表示每个小块变换后的维度大小。
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        ##[batch_size, channels, _height * _width, self.stride * self.stride] --> [batch_size, self.stride * self.stride, channels, _height, _width]
        x = x.view(batch_size, -1, _height, _width)
        ##(stride * stride) 和 channels 维度合并为一个 -1 维度

        return x
