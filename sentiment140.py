import csv
import os
import re

if not os._exists('sentiment140/positive'):
    os.mkdir('sentiment140/positive')
if not os._exists('sentiment140/negative'):
    os.mkdir('sentiment140/negative')

with open('sentiment140/training.1600000.processed.noemoticon.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    with open('sentiment140/negative.txt', "w") as negfile:
        with open('sentiment140/positive.txt', "w") as posfile:
            for row in reader:
                content = re.sub(r"\n+", ' ', str(row[5]))
                if row[0] == "0":
                    negfile.write(content)
                if row[0] == '4':
                    posfile.write(content)
