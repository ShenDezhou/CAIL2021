import os

from PIL import Image
from resizeimage import resizeimage
import joblib

def preprocess(FOLDER, output_file="exam.data"):
    test_l = {
        'data': [],
        'labels': [],
        'filenames': []
    }
    original_files = []
    for root, folders, names in os.walk(FOLDER):
        # for _, _, names in os.walk(root):
        counter = 0
        for file in names:
            filename = os.path.join(root, file)
            # newfilename = file[:23]+"_"+str(random.randint(1,130))+"_"+str(counter)+".png"
            counter += 1
            # with open(filename, 'r') as fd_img:
            img = Image.open(filename)
            if img.format != "png":
                img = img.convert('RGB')
            img = resizeimage.resize_cover(img, [32, 32])
            arr = list(img.tobytes())
            original_files.append(file)
            test_l['data'].append(arr)
            test_l['labels'].append(0)
            test_l['filenames'].append(filename)

    joblib.dump(test_l, os.path.join(root, output_file), compress=3)

    return original_files


def read_joblib(filename):

    test = joblib.load(filename)

    return test['data'], test["filenames"], test["labels"]
# print('FIN')
