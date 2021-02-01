import argparse
import json
import os

import torch
from PIL import Image
from resizeimage import resizeimage
from torch import nn
from tqdm import tqdm

from torchocr.networks import build_model
from torchocr.datasets.RecDataSet import RecDataProcess
from torchocr.utils import CTCLabelConverter, CTCAsWholeLabelConverter
import cv2
LABELS = range(1,131)

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


    def predict_top5(self, img):
        # 预处理根据训练来
        img = self.process.resize_with_specific_height(img)
        # img = self.process.width_pad_img(img, 120)
        img = self.process.normalize_img(img)
        tensor = torch.from_numpy(img.transpose([2, 0, 1])).float()
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        out = self.model(tensor)
        out = out.softmax(dim=2)#.detach().cpu().numpy()
        # answer = torch.topk(out, 5, dim=-1)[1]  # return indices
        # answer_label = [LABELS[l] for l in answer]  # the competetion system start with 1
        answer_label = self.converter.decode_top5(out)
        return answer_label


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
    parser.add_argument('--outfile', required=False, type=str, help='output path for predict',
                        default='output/result.json')
    args = parser.parse_args()
    model = RecInfer(args.model_path)


    for root, folders, core_names in os.walk(args.img_path):
        if False:
            for name in core_names:
                # img = cv2.imread(os.path.join(root, name))
                img = Image.open(os.path.join(root, name))
                if img.format != "png":
                    img = img.convert('RGB')
                newfilename = os.path.join("temp", name.split('.')[0]+".png")
                img = resizeimage.resize_cover(img, [32, 32])
                img.save(newfilename, 'png')


    answer_list = []
    for root, folders, names in os.walk("temp"):
        for name in names:
            img = cv2.imread(os.path.join(root, name))
            out = model.predict_top5(img)
            answer_list.extend(out[0])
            print(out[0])

    int_answer_list = []
    for answer in answer_list:
        answer = [int(i) for i in answer]
        int_answer_list.append(answer)
    print(int_answer_list)
    # 4. Write answers to file
    # id_list = pd.read_csv(in_file)['id'].tolist()
    pred_result = dict(zip(core_names, int_answer_list))

    with open(args.outfile, 'w') as fout:
        json.dump(pred_result, fout, ensure_ascii=False, indent=4)
    print("FIN")
