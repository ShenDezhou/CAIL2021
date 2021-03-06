import argparse
import json
import os

import torch
from torch import nn
from torchocr.networks import build_model
from torchocr.datasets.RecDataSet import RecDataProcess
from torchocr.utils import CTCLabelConverter, CTCAsWholeLabelConverter
import cv2

class RecInfer:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        self.model = build_model(cfg.model)
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        self.model.load_state_dict(state_dict)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.process = RecDataProcess(cfg.dataset.train.dataset)
        self.converter = CTCAsWholeLabelConverter(cfg.dataset.alphabet)

    def predict(self, img):
        # 预处理根据训练来
        img = self.process.resize_with_specific_height(img)
        # img = self.process.width_pad_img(img, 120)
        img = self.process.normalize_img(img)
        tensor = torch.from_numpy(img.transpose([2, 0, 1])).float()
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        out = self.model(tensor)
        txt = self.converter.decode(out.softmax(dim=2).detach().cpu().numpy())
        return txt


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='PytorchOCR infer')
    parser.add_argument('--model_path', required=False, type=str, help='rec model path', default=r'F:\CAIL\CAIL2020\cocr\model\CRNN\checkpoint\latest.pth')
    parser.add_argument('--img_path', required=False, type=str, help='img path for predict', default=r'F:\CAIL\CAIL2020\cocr\data\icdar2015\recognition\test\img_2_0.jpg')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # ===> 获取配置文件参数
    # cfg = parse_args()
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--config', type=str, default='config/rec.json', help='train config file path')
    parser.add_argument('--model_path', required=False, type=str, help='rec model path', default=r'F:\CAIL\CAIL2020\cocr\model\rec-model.bin')
    parser.add_argument('--img_path', required=False, type=str, help='img path for predict', default=r'F:\CAIL\CAIL2021\dogfl\test\TEST_A')
    parser.add_argument('--outfile', required=False, type=str, help='img path for predict',
                        default='output/result.json')
    args = parser.parse_args()
    model = RecInfer(args.model_path)

    answer_list = []
    for root, folders, names in os.walk(args.img_path):
        for name in names:
            img = cv2.imread(os.path.join(root, name))
            model = RecInfer(args.model_path)
            out = model.predict(img)
            answer_list.append(out)
            print(out)

    print(answer_list)
    # 4. Write answers to file
    # id_list = pd.read_csv(in_file)['id'].tolist()
    pred_result = dict(zip(names, answer_list))

    with open(args.outfile, 'w') as fout:
        json.dump(pred_result, fout, ensure_ascii=False, indent=4)
    print("FIN")
