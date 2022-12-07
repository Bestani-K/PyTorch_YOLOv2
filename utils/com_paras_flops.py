import torch
from thop import profile


def FLOPs_and_Params(model, img_size, device): ##计算模型的浮点操作数和参数量的函数
    x = torch.randn(1, 3, img_size, img_size).to(device)
    print('==============================')
    flops, params = profile(model, inputs=(x, ))
    print('==============================')
    print('FLOPs : {:.2f} B'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))


if __name__ == "__main__":
    pass
