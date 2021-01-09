import json

import pandas

for filename in ['bmes_train.json','bmes_test.json']:
    data_frame = json.load(open(filename, 'r', encoding='utf-8'))
    for item in data_frame:
        if "entities" in item.keys():
            item.pop('entities')
        item.pop('id')
    df = pandas.DataFrame(data_frame)
    df.to_csv(filename+'.csv', index=False)

print('FIN')
