import os
import cv2
import unicodedata

def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

ratio = []
with open('image_train.csv','w', encoding='utf-8') as f:
    for root, folders, names in os.walk('reshape/1'):
        count=0
        for file in names:
            label = file.split('_')[-2]
            imgdata = cv2.imread(os.path.join(root, file))
            img_channels = imgdata.transpose(2,0,1)
            for i in range(3):
                img_size = len(img_channels[i].tostring())
                img = img_channels[i].tostring().decode(encoding="utf-8", errors="ignore")
                actrual_size = len(img.encode(encoding='utf-8'))
                encoderatio = actrual_size * 1.0 / img_size
                img = remove_control_characters(img).strip()
                if len(img) > 10:
                    ratio.append(encoderatio)
                    f.write(img)
                    f.write('\n')
                    count+= 1
print(len(ratio), min(ratio), max(ratio))

ratio = []
with open('image_eval.csv','w', encoding='utf-8') as f:
    for root, folders, names in os.walk('reshape/2'):
        count=0
        for file in names:
            label = file.split('_')[-2]
            imgdata = cv2.imread(os.path.join(root, file))
            img_channels = imgdata.transpose(2, 0, 1)
            for i in range(3):
                img_size = len(img_channels[i].tostring())
                img = img_channels[i].tostring().decode(encoding="utf-8", errors="ignore")
                actrual_size = len(img.encode(encoding='utf-8'))
                encoderatio = actrual_size * 1.0 / img_size
                img = remove_control_characters(img).strip()
                if len(img) > 10:
                    ratio.append(encoderatio)
                    f.write(img)
                    f.write('\n')
                    count+= 1
print(len(ratio), min(ratio), max(ratio))

