# SPDX-License-Identifier: BSD-3-Clause

from numba import cuda
from math import ceil
import numpy as np


# Read expense report.
with open('input') as f:
    entries = np.array([int(s) for s in f.readlines()], dtype=np.int32)


# Part 1 - a brute force search of all pairs using one thread per pair.
@cuda.jit
def find_2020(entries, answer):
    # Get the pair for this thread
    i, j = cuda.grid(2)

    # Ensure we don't read off the end of the array there are more threads than
    # entries in a grid dimension. Also avoid duplicating the pairs when (i, j)
    # == (j, i)
    n_entries = len(entries)
    if i < n_entries and j < n_entries and (i <= j + 1):
        pair_sum = entries[i] + entries[j]
        if pair_sum == 2020:
            # Use an atomic store just in case more than one pair writes the
            # solution (unlikely to be necessary, especially with the guard
            # above, but added just to be on the safe side)
            cuda.atomic.exch(answer, 0, entries[i] * entries[j])


# 32 * 32 (1024) threads is a reasonable, arbitrarily chosen block size. We use
# a 2D grid because it maps nicely on to the 2D problem space of (n_entries,
# n_entries)
n_entries = len(entries)
n_threads = (32, 32)
# How many blocks do we need for the given number of entries?
n_blocks = (ceil(n_entries / n_threads[0]), ceil(n_entries / n_threads[1]))

# We can't return values from kernels, so allocate space for the result and
# launch the kernel.
answer = np.zeros(1, dtype=np.int32)
find_2020[n_blocks, n_threads](entries, answer)

print(f"The answer to part 1 is {answer[0]}")


# Part 2 - a further brute force search in a similar vein to part 1.
@cuda.jit
def find_2020_part2(entries, answer):
    i, j, k = cuda.grid(3)
    n_entries = len(entries)
    # I couldn't be bothered to work out the condition to check each triple
    # exactly once, so it is omitted here.
    if i < n_entries and j < n_entries and k < n_entries:
        triple_sum = entries[i] + entries[j] + entries[k]
        if triple_sum == 2020:
            # Multiple threads will definitely write the answer because of the
            # lack of guard condition, so the atomic exchange here does avoid
            # undefined behaviour (although even using a normal assignment we'd
            # be unlikely to observe anything unexpected in practice).
            cuda.atomic.exch(answer, 0, entries[i] * entries[j] * entries[k])


# 8 * 8 * 8 (512) threads makes for a nice cubic block size that is not too
# large.
n_threads = (8, 8, 8)
# Compute n_threads using a similar strategy to the above.
n_blocks = (
    ceil(n_entries / n_threads[0]),
    ceil(n_entries / n_threads[1]),
    ceil(n_entries / n_threads[2])
)

# Allocate result and compute answer.
answer = np.zeros(1, dtype=np.int32)
find_2020_part2[n_blocks, n_threads](entries, answer)

print(f"The answer to part 2 is {answer[0]}")
