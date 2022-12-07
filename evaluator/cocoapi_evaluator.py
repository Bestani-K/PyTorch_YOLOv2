import json
import tempfile
import torch
import numpy as np
from pycocotools.cocoeval import COCOeval

from data.coco import COCODataset


class COCOAPIEvaluator():
    def __init__(self, data_dir, img_size, device, testset=False, transform=None):
        self.img_size = img_size
        self.transform = transform
        self.device = device
        self.map = -1.

        self.testset = testset
        if self.testset:
            json_file='image_info_test-dev2017.json'
            image_set = 'test2017'
        else:
            json_file='instances_val2017.json'
            image_set='val2017'

        self.dataset = COCODataset(
            data_dir=data_dir,
            img_size=img_size,
            json_file=json_file,
            transform=None,
            image_set=image_set)


    def evaluate(self, model):
        model.eval()
        ids = []
        data_dict = []
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))

        # start testing
        for index in range(num_images): # all the data in val2017
            if index % 500 == 0:
                print('[Eval: %d / %d]'%(index, num_images))

            img, id_ = self.dataset.pull_image(index)  # load a batch
            if self.transform is not None:
                x = torch.from_numpy(self.transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(self.device)
            scale = np.array([[img.shape[1], img.shape[0],  ##[[w, h, w, h]]
                            img.shape[1], img.shape[0]]])
            
            id_ = int(id_)
            ids.append(id_)
            with torch.no_grad():  ##避免在预测时占用显存
                outputs = model(x)
                bboxes, scores, labels = outputs
                bboxes *= scale  ##转换为图像的实际坐标
            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                label = self.dataset.class_ids[int(labels[i])]
                
                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i]) # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score} # COCO json format
                data_dict.append(A)

        annType = ['segm', 'bbox', 'keypoints']  ##分割标注、边界框标注和关键点标注的评估

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            if self.testset:
                json.dump(data_dict, open('yolo_2017.json', 'w'))
                cocoDt = cocoGt.loadRes('yolo_2017.json')
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, 'w'))
                cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            ##cocoDt是用模型预测生成的结果数据，它也是COCO格式的，annType[1]指定评估的注释类型，这里是bbox，表示边界框。
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()  ##检测结果的性能
            cocoEval.accumulate()
            cocoEval.summarize()  ##汇总结果并输出评估指标

            ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]  ##评估结果保存在self.ap50_95和self.ap50中
            print('ap50_95 : ', ap50_95)
            print('ap50 : ', ap50)
            self.map = ap50_95
            self.ap50_95 = ap50_95
            self.ap50 = ap50

            return ap50_95, ap50
        else:
            return 0, 0

