import json

with open('result.json','r') as f:
    js = json.load(f)
    tot = len(js)
    cor = 0
    for k,v in js.items():
        if v[0] == 128:
           cor += 1

    print('ACC:', cor*1.0/tot)
print('FIN')