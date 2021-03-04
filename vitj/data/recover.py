import os

import cv2
import joblib
from PIL import Image
import numpy

data_pack = joblib.load("test.data")

# train_l = {
#     'data':[],
#     'labels':[],
#     'filenames':[]
# }

data = data_pack['data']
labels = data_pack['labels']
filenames = data_pack["filenames"]

_ = [os.makedirs("cifar10_test/"+str(label), exist_ok=True) for label in set(labels)]
for i in range(len(data_pack['data'])):
    d = data[i].reshape(3,32,32)
    d = d.transpose((1,2,0)).copy()
    pixels = Image.frombytes('RGB', (32,32), d, 'raw')

    filename = str(filenames[i].decode("utf-8"))
    format = filename.split('.')[-1]
    filename = filename.replace("_s_", "_")
    pixels.save(f"cifar10_test/{labels[i]}/" + filename, format)
    #print(labels[i])
print('FIN')