import argparse
import csv


parser = argparse.ArgumentParser(description="Generate label file from text file")
parser.add_argument("--text_file_path", type=str, help="path of target text file")
parser.add_argument("--csv_file_path", type=str, help="path of csv file to save labels")

args = parser.parse_args()

print("get label from", args.text_file_path, "...")

labels = {}
with open(args.text_file_path, "r") as f:
    for line in f.readlines():
        line = line.replace("\n", "")
        for char in line:
            if char not in labels:
                labels[char] = 1
            else:
                labels[char] += 1

labels = dict(reversed(sorted(labels.items(), key=lambda item: item[1])))

id2char = {0: "<pad>", 1: "<sos>", 2: "<eos>", 2000: "<mask>"}

for i, label in enumerate(labels):
    if i+3 == 2000:
        break
    id2char[i+3] = label
    
print("save label to", args.csv_file_path)
    
with open(args.csv_file_path, "w") as f:
    wr = csv.writer(f)
    wr.writerow(["id", "char"])
    for k, v in id2char.items():
        wr.writerow([k, v])
        
