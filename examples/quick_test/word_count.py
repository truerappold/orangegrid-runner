"""Toy analysis: count words in one text file, write <output>/counts.csv.

    python word_count.py INPUT_FILE OUTPUT_DIR

Stands in for any real analysis script: it takes an input and an output
folder on the command line, which is all ogrun needs.
"""
import csv
import os
import sys
from collections import Counter

src, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
words = Counter(w.strip('.,;:!?"()').lower() for w in open(src).read().split())
with open(os.path.join(out_dir, 'counts.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['file', 'word', 'count'])
    for word, n in words.most_common():
        if word:
            w.writerow([os.path.basename(src), word, n])
print(f'{src}: {sum(words.values())} words, {len(words)} distinct')
