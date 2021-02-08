import json
import sys
sys.path.extend("..")
import lawa
import os
import argparse

frequency = {}
word2id = {"PAD": 0, "UNK": 1}
min_freq = 10


def cut(s):
    arr = list(lawa.cut(s))
    for word in arr:
        if word not in frequency:
            frequency[word] = 0
        frequency[word] += 1

    return arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/")
    parser.add_argument('--output', default="dataprocess/")
    parser.add_argument('--gen_word2id', action="store_true", default=True)
    args = parser.parse_args()

    input_path = args.data
    output_path = args.output

    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if filename in output_path:
            continue
        fin = open(os.path.join(input_path, filename), "r", encoding="utf8")
        fout = open(os.path.join(output_path, filename), "w", encoding="utf8")

        all_data = json.load(fin)
        for data in all_data:
            data["Content"] = cut(data["Content"])
            for questions in data["Questions"]:
                if isinstance(questions, list):
                    for question in questions:
                        question["Question"] = cut(question["Question"])
                        for choice in question["Choices"]:
                            choice = cut(choice)
                else:
                    question = questions
                    question["Question"] = cut(question["Question"])
                    for choice in question["Choices"]:
                        choice = cut(choice)


        json.dump(all_data,fout,  ensure_ascii=False, sort_keys=True, indent=4)

    if args.gen_word2id:
        for word in frequency:
            if frequency[word] >= min_freq:
                word2id[word] = len(word2id)
        json.dump(word2id, open("data/word2id.txt", "w", encoding="utf8"), indent=2, ensure_ascii=False)
