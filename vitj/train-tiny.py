import os

import jittor as jt
import jittor.nn as nn 
from dataset import TsinghuaDog
from jittor import transform
from jittor.optim import Adam, SGD
from tqdm import tqdm
import numpy as np
from model import Net
import argparse 



jt.flags.use_cuda=1

def get_path(path):
    """Create the path if it does not exist.

    Args:
        path: path to be used

    Returns:
        Existed path
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path

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


best_acc = -1.0
def evaluate(model, val_loader, epoch=0, save_path='./best_model.bin'):
    model.eval()
    global best_acc
    total_acc = 0
    total_num = 0
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [EVAL]')
    for images, labels in pbar:
        output = model(images)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]
        pbar.set_description(f'Epoch {epoch} [EVAL] acc = {total_acc / total_num :.2f}')
    acc = total_acc / total_num 
    if acc > best_acc:
        best_acc = acc
        get_path(save_path)
        model.save(save_path)
    print ('Test in epoch', epoch, 'Accuracy is', acc, 'Best accuracy is', best_acc)
#python train-tiny.py --epochs 5 --batch_size 32 --dataroot /mnt/data/dogfldocker --model_path model/res50/model.bin --resume False
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=130)

    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)

    parser.add_argument('--dataroot', type=str, default='/content/drive/MyDrive/dogflg/data2/')
    parser.add_argument('--model_path', type=str, default='./best_model.bin')

    parser.add_argument('--sampleratio', type=float, default=0.8)

    args = parser.parse_args()
    
    transform_train = transform.Compose([
        transform.Resize((256, 256)),
        transform.CenterCrop(224),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize(0.485, 0.229),
        # transform.ImageNormalize(0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    root_dir = args.dataroot
    train_loader = TsinghuaDog(root_dir, batch_size=args.batch_size, train=True, part='train', shuffle=True, transform=transform_train, sample_rate=args.sampleratio)

    transform_test = transform.Compose([
        transform.Resize((256, 256)),
        transform.CenterCrop(224),
        transform.ToTensor(),
        transform.ImageNormalize(0.485, 0.229),
        # transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_loader = TsinghuaDog(root_dir, batch_size=args.batch_size, train=False, part='val', shuffle=False, transform=transform_test, sample_rate=args.sampleratio)

    epochs = args.epochs
    model = Net(num_classes=args.num_classes)
    lr = args.lr
    weight_decay = args.weight_decay
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.99)
    if args.resume:
        model.load(args.model_path)
        print('model loaded', args.model_path)

    #random save for test
    #model.save(args.model_path)
    if args.eval:
        evaluate(model, val_loader, save_path=args.model_path)
        return 
    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch)
        evaluate(model, val_loader, epoch, save_path=args.model_path)


if __name__ == '__main__':
    main()

