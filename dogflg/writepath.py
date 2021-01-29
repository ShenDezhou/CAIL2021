import os


for root, folders, names in os.walk(r'F:\CAIL\CAIL2021\dogflg\data\reshape'):
    for folder in folders:
        result = []
        for _, _, names in os.walk(os.path.join(root,folder)):
            result = [os.path.join(root, folder, name)+"\n" for name in names]
            types = [int(line.split('_')[-2]) for line in result]
            result_file = 'data/train.txt' if folder == "1" else 'data/val.txt'
            with open('data/train.txt','w') as fw:
                for i in range(len(result)):
                    fw.write(result[i]+'\t'+types[i])
            if folder == "2":
                with open('data/val.txt','w') as fw:
                    fw.writelines(result)
print("FIN")


