import json

import jittor as jt
import jittor.nn as nn 
from dataset import TsinghuaDogExam
from jittor import transform
from tqdm import tqdm
import numpy as np
from model import Net
import argparse 


jt.flags.use_cuda=0

def train(model, train_loader, optimizer, epoch):
    model.train() 
    total_acc = 0
    total_num = 0
    losses = 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    for images, labels in pbar:
        output = model(images)
        loss = nn.cross_entropy_loss(output, labels)
        optimizer.step(loss) 
        pred = np.argmax(output.data, axis=1)
        acc = np.mean(pred == labels.data) * 100 
        total_acc += acc
        total_num += labels.shape[0] 
        losses += loss
        pbar.set_description(f'Epoch {epoch} [TRAIN] loss = {loss.data[0]:.2f}, acc = {acc:.2f}')

def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort

def evaluate(model, val_loader):
    model.eval()

    answer_list = []
    for images, labels in tqdm(val_loader):
        output = model(images)

        _, pred = topk_(output.data, 5, axis=1)

        print(pred)
        answer_list.extend(pred)
    answer_list = [nd.tolist() for nd in answer_list]
    # label start from 1
    for i in range(len(answer_list)):
        answer_list[i] = [i + 1 for i in answer_list[i]]
    return answer_list



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=130)

    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--eval', type=bool, default=False)

    parser.add_argument('--dataroot', type=str, default='/content/drive/MyDrive/dogfl/data/TEST_A/')
    parser.add_argument('--model_path', type=str, default='./best_model.bin')
    parser.add_argument('--out_file', type=str, default='./result.json')


    args = parser.parse_args()

    root_dir = args.dataroot

    transform_test = transform.Compose([
        transform.Resize((512, 512)),
        transform.CenterCrop(448),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_loader = TsinghuaDogExam(root_dir, batch_size=16, train=False, list_path= "/content/drive/MyDrive/dogflj/data/test_a.lst", shuffle=False, transform=transform_test)

    model = Net(num_classes=args.num_classes)
    if args.resume:
        model.load(args.model_path)

    name_list = []
    with open("/content/drive/MyDrive/dogflj/data/test_a.lst", 'r') as f:
        for line in f:
            name_list.append(line.strip())

    top5_class_list = evaluate(model, val_loader)
    # label start from 1, however it doesn't
    pred_result = dict(zip(name_list, top5_class_list))

    with open(args.out_file, 'w') as fout:
        json.dump(pred_result, fout, ensure_ascii=False, indent=4)




if __name__ == '__main__':
    main()

