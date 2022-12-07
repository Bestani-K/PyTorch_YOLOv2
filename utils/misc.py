import torch
import numpy as np


def detection_collate(batch): ##数据加载
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def load_weight(model, path_to_ckpt=None): ##加载权重文件
    # check
    if path_to_ckpt is None:
        print('no weight file ...')
        return model
        
    checkpoint_state_dict = torch.load(path_to_ckpt, map_location='cpu')
    # model state dict
    model_state_dict = model.state_dict()
    # check
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    print('Finished loading model!')

    return model
