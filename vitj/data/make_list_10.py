import os

ROOT = 'cifar10/cifar10_train/'
with open('cifar10/TrainAndValList/train.lst','w') as fw:
    for root, folders, filenames in os.walk(ROOT):
        for folder in folders:
            for _, _, names in os.walk(os.path.join(ROOT, folder)):
                for name in names:
                    fw.write(os.path.join(ROOT, folder, name)+'\n')

ROOT = 'cifar10/cifar10_test/'
with open('cifar10/TrainAndValList/validation.lst', 'w') as fw:
    for root, folders, filenames in os.walk(ROOT):
        for folder in folders:
            for _, _, names in os.walk(os.path.join(ROOT, folder)):
                for name in names:
                    fw.write(os.path.join(ROOT, folder, name)+'\n')

print('FIN')