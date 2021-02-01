import os
from PIL import Image
from resizeimage import resizeimage

RESHAPE='reshape'

train_list = []
val_list = []
with open("TrainAndValList/train.lst",'r', encoding='utf-8') as f:
    for line in f:
        train_list.append(line.lstrip(".//").strip())

with open("TrainAndValList/validation.lst",'r', encoding='utf-8') as f:
    for line in f:
        val_list.append(line.lstrip(".//").strip())

def train_or_val(folder, file):
    path = os.path.join(folder,file)
    path = path.replace("\\","/")
    if path in train_list:
        return 1
    assert path in val_list, path
    return 2

for root, folders, names in os.walk('low-resolution'):
    for folder in folders:
        for _, _, names in os.walk(os.path.join(root,folder)):
            size, type_id, type_name = folder.split('-')
            # type_id = int(type_id[1:])
            counter = 0
            for file in names:
                filename = os.path.join(root, folder, file)
                trainvalid = train_or_val(folder, file)
                newfilename = os.path.join(RESHAPE, str(trainvalid), type_name[:23]+"_"+type_id[-3:]+"_"+str(counter)+".png")
                counter += 1
                # with open(filename, 'r') as fd_img:
                img = Image.open(filename)
                if img.format !="png":
                    img = img.convert('RGB')
                img = resizeimage.resize_cover(img, [32, 32])
                img.save(newfilename, 'png')

print('FIN')
