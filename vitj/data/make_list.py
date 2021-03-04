import os

ROOT = 'cifar100/cifar100_train/'
with open('cifar100/TrainAndValList/train.lst','w') as fw:
    for root, folders, filenames in os.walk(ROOT):
        for folder in folders:
            for _, _, names in os.walk(os.path.join(ROOT, folder)):
                for name in names:
                    fw.write(os.path.join(ROOT, folder, name)+'\n')

ROOT = 'cifar100/cifar100_test/'
with open('cifar100/TrainAndValList/validation.lst', 'w') as fw:
    for root, folders, filenames in os.walk(ROOT):
        for folder in folders:
            for _, _, names in os.walk(os.path.join(ROOT, folder)):
                for name in names:
                    fw.write(os.path.join(ROOT, folder, name)+'\n')

print('FIN')