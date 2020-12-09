# SPDX-License-Identifier: BSD-3-Clause

from numba import cuda
from math import ceil
import numpy as np


USE_EXAMPLE = False

if USE_EXAMPLE:
    filename = 'example_input'
    preamble_length = 5
else:
    filename = 'input'
    preamble_length = 25

with open(filename) as f:
    seq = np.asarray([int(x.strip()) for x in f.readlines()], dtype=np.uint32)


@cuda.jit
def find_impostor(seq, answer):
    i = cuda.grid(1)
    lower = i
    upper = lower + preamble_length

    if upper >= len(seq):
        return

    current = seq[upper]

    found = False
    for xi in range(lower, upper):
        for yi in range(lower, upper):
            x = seq[xi]
            y = seq[yi]

            if x == y:
                continue
            if x + y == current:
                found = True

    if not found:
        cuda.atomic.min(answer, 0, upper)


n_threads = 256
n_blocks = ceil(len(seq) / n_threads)

impostor_idx = np.asarray([len(seq) + 1], dtype=np.uint32)
find_impostor[n_blocks, n_threads](seq, impostor_idx)
impostor = seq[impostor_idx[0]]
print(f"The answer to part 1 is {impostor}")


@cuda.jit
def find_contiguous(impostor_idx, seq, answer):
    i = cuda.grid(1)
    if i >= impostor_idx:
        return

    for j in range(impostor_idx - i):
        candidate_sum = 0
        candidate_min = 999999999999999
        candidate_max = 0
        for k in range(i, i+j):
            current = seq[k]
            candidate_sum += current
            candidate_min = min(candidate_min, current)
            candidate_max = max(candidate_max, current)

        if candidate_sum == impostor:
            answer[0] = candidate_min + candidate_max


answer = np.zeros(1, dtype=np.uint32)
n_blocks = ceil(impostor_idx / n_threads)
find_contiguous[n_blocks, n_threads](impostor_idx[0], seq, answer)
print(f"The answer to part 2 is {answer[0]}")
