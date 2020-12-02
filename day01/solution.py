from numba import cuda
from math import ceil
import numpy as np


with open('input') as f:
    entries = np.array([int(s) for s in f.readlines()], dtype=np.int32)


@cuda.jit
def find_2020(entries, answer):
    i, j = cuda.grid(2)
    n_entries = len(entries)
    if i < n_entries and j < n_entries and (i <= j + 1):
        pair_sum = entries[i] + entries[j]
        if pair_sum == 2020:
            cuda.atomic.exch(answer, 0, entries[i] * entries[j])


n_entries = len(entries)
n_threads = (32, 32)
n_blocks = (ceil(n_entries / n_threads[0]), ceil(n_entries / n_threads[1]))

answer = np.zeros(1, dtype=np.int32)
find_2020[n_blocks, n_threads](entries, answer)

print(f"The answer to part 1 is {answer[0]}")


@cuda.jit
def find_2020_part2(entries, answer):
    i, j, k = cuda.grid(3)
    n_entries = len(entries)
    if i < n_entries and j < n_entries and k < n_entries:
        triple_sum = entries[i] + entries[j] + entries[k]
        if triple_sum == 2020:
            cuda.atomic.exch(answer, 0, entries[i] * entries[j] * entries[k])


n_threads = (8, 8, 8)
n_blocks = (
    ceil(n_entries / n_threads[0]),
    ceil(n_entries / n_threads[1]),
    ceil(n_entries / n_threads[2])
)

answer = np.zeros(1, dtype=np.int32)
find_2020_part2[n_blocks, n_threads](entries, answer)

print(f"The answer to part 2 is {answer[0]}")
