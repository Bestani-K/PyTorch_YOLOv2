import torch
import torch.nn as nn
import os


model_urls = {
    "darknet19": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet19.pth",
}


__all__ = ['darknet19']


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class DarkNet_19(nn.Module):
    def __init__(self):        
        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            ##输入通道数, 输出通道数, 卷积核大小, 步长
            nn.MaxPool2d((2,2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2,2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2,2), 2)
        )

        # output : stride = 8, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
        )
        
        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )

    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, 3, H, W]
        Output:
            output: (Dict) {
                'c3': c3 -> Tensor[B, C3, H/8, W/8],
                'c4': c4 -> Tensor[B, C4, H/16, W/16],
                'c5': c5 -> Tensor[B, C5, H/32, W/32],
            }
        """
        c1 = self.conv_1(x)                    # [B, C1, H/2, W/2]
        c2 = self.conv_2(c1)                   # [B, C2, H/4, W/4]
        c3 = self.conv_3(c2)                   # [B, C3, H/8, W/8]
        c3 = self.conv_4(c3)                   # [B, C3, H/8, W/8]
        c4 = self.conv_5(self.maxpool_4(c3))   # [B, C4, H/16, W/16]
        c5 = self.conv_6(self.maxpool_5(c4))   # [B, C5, H/32, W/32]

        output = {
            'c3': c3,
            'c4': c4,
            'c5': c5
        }
        return output


def build_darknet19(pretrained=False):
    # model
    model = DarkNet_19()
    feat_dims = [256, 512, 1024]  ##对应于3个卷积层输出特征图的通道数

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['darknet19']
        # checkpoint state dict
        checkpoint_state_dict = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # model state dict
        model_state_dict = model.state_dict() ##用于保存和加载模型的训练状态
        # check
        for k in list(checkpoint_state_dict.keys()): ##确保在加载预训练模型时，只有形状和名称都匹配的参数才会被加载
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model, feat_dims


if __name__ == '__main__':
    import time
    model, feats = build_darknet19(pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for k in outputs.keys():
        print(outputs[k].shape)
