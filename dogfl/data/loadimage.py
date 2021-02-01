import io
import os

import cv2
import joblib

from PIL import Image

train_l = {
    'data':[],
    'labels':[],
    'filenames':[]
}

test_l = {
    'data':[],
    'labels':[],
    'filenames':[]
}


for root, folders, names in os.walk('reshape/1'):
    count=0
    for file in names:
        label = file.split('_')[-2]
        # filename = os.path.join(root,file)
        # img = Image.open(filename)
        # arr = list(img.tobytes())
        img = cv2.imread(os.path.join(root, file))

        train_l['data'].append(img)
        train_l['labels'].append(int(label))
        train_l['filenames'].append(file)
        count+= 1

for root, folders, names in os.walk('reshape/2'):
    count=0
    for file in names:
        label = file.split('_')[-2]
        # filename = os.path.join(root,file)
        # img = Image.open(filename)
        # arr = list(img.tobytes())
        img = cv2.imread(os.path.join(root, file))

        test_l['data'].append(img)
        test_l['labels'].append(int(label))
        test_l['filenames'].append(file)
        count+= 1



joblib.dump(train_l,"train.data", compress=3)
joblib.dump(test_l,"test.data", compress=3)
print('FIN')