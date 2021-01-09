import json
import numpy
test_json = json.load(open('bmes_test.json', 'r', encoding='utf-8'))

ll = [len(item['text']) for item in test_json]


for i in range(21):
    p = numpy.percentile(ll, i*5)
    print(p)

# 1~134, 104 is 95%(train)
# 12~133, 100 is 95%(test)