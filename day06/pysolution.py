# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

with open('input') as f:
    batch = f.read()

groups = [[line.strip() for line in group.split()]
          for group in batch.split('\n\n')]

total = 0

for group in groups:
    answers = np.zeros(26, dtype=np.bool_)
    for person in group:
        for answer in person:
            idx = ord(answer) - ord('a')
            answers[idx] = True
    total += answers.sum()

print(f'The answer to part 1 is {total}')

total = 0

for group in groups:
    answers = [{answer for answer in person} for person in group]
    common_yeses = answers[0].intersection(*answers[1:])
    total += len(common_yeses)

print(f'The answer to part 2 is {total}')
