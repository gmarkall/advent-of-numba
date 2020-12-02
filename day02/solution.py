from math import ceil
from numba import cuda
import numpy as np


def parse(line):
    # Maybe a regex would be better, but I am rubbish at those.
    lower = int(line.split('-')[0])
    upper = int(line.split('-')[1].split()[0])
    letter = ord(line.split()[1][0])
    password = [ord(x) for x in line.split()[2]]
    length = len(password)
    return lower, upper, letter, length, password


lowers, uppers, letters, lengths, passwords = [], [], [], [], []
with open('input') as f:
    for line in f.readlines():
        lower, upper, letter, length, password = parse(line)
        lowers.append(lower)
        uppers.append(upper)
        letters.append(letter)
        lengths.append(length)
        passwords.append(password)

max_length = max(lengths)
for i in range(len(passwords)):
    if len(passwords[i]) < max_length:
        passwords[i].extend([0] * (max_length - len(passwords[i])))

dtype = [
    ('lower', '<i4'),
    ('upper', '<i4'),
    ('letter', '<i4'),
    ('length', '<i4'),
    ('password', '<i4', max(lengths))
]

entries = np.array(list(zip(lowers, uppers, letters, lengths, passwords)),
                   dtype=dtype)


@cuda.jit
def count_valid_passwords(entries, answer):
    i = cuda.grid(1)
    if i < len(entries):
        entry = entries[i]
        count = 0
        letter = entry['letter']
        password = entry['password']
        for p in range(entry['length']):
            if password[p] == letter:
                count += 1
        if count >= entry['lower'] and count <= entry['upper']:
            cuda.atomic.inc(answer, 0, len(entries))


n_threads = 256
n_blocks = ceil(len(entries) / n_threads)

answer = np.zeros(1, dtype=np.uint32)
count_valid_passwords[n_blocks, n_threads](entries, answer)
print(f"The answer to part 1 is {answer[0]}")


@cuda.jit
def count_valid_new_policy(entries, answer):
    i = cuda.grid(1)
    if i < len(entries):
        entry = entries[i]
        p1 = entry['password'][entry['lower'] - 1]
        p2 = entry['password'][entry['upper'] - 1]
        letter = entry['letter']
        matches = int(p1 == letter) + int(p2 == letter)
        if matches == 1:
            cuda.atomic.inc(answer, 0, len(entries))


answer = np.zeros(1, dtype=np.uint32)
count_valid_new_policy[n_blocks, n_threads](entries, answer)
print(f"The answer to part 2 is {answer[0]}")
