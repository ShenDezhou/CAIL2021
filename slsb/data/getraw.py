import json

with open('test.json','r', encoding='utf8') as f:
    for line in f:
        d = json.loads(line)
        lines = d['text']
        lines = [t['sentence'] for t in lines]
        jline = r"\n".join(lines)
        print(jline)