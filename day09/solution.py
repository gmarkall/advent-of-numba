# SPDX-License-Identifier: BSD-3-Clause

# The pure Python version rewritten to use CUDA with Numba. Various
# modifications are needed to implement Python functions / features not present
# in the CUDA - these are noted in the comments below.

from numba import cuda
from math import ceil
import numpy as np


# The example and real inputs have different preamble lengths

USE_EXAMPLE = False

if USE_EXAMPLE:
    filename = 'example_input'
    preamble_length = 5
else:
    filename = 'input'
    preamble_length = 25

# Read in input

with open(filename) as f:
    # lists have very little support in the CUDA target - use a NumPy array for
    # the input instead
    seq = np.asarray([int(x.strip()) for x in f.readlines()], dtype=np.uint32)


# Part 1

# The kernel for part 1 parallelises the search for the impostor by assigning
# each thread its own window to check.
@cuda.jit
def find_impostor(seq, answer):
    # Thread ID
    i = cuda.grid(1)
    # Compute the window for this thread
    lower = i
    upper = lower + preamble_length

    # If the window is outside the input, then don't do anything. This "early
    # return" replaces the while loop in the sequential version
    if upper >= len(seq):
        return

    # The potential impostor's value
    current = seq[upper]

    found = False
    # Itertools is not supported in CUDA, so we implement the product operation
    # with nested loops and index into the sequence using the indices we
    # generate.
    for xi in range(lower, upper):
        for yi in range(lower, upper):
            x = seq[xi]
            y = seq[yi]

            # Similar checks as found in the sequential version
            if x == y:
                continue
            if x + y == current:
                # No need to increment any indices here, because other threads
                # handle other indices
                found = True

    if not found:
        # We're looking for the first number that's not a sum of any pair in its
        # preamble - as there may be multiple numbers like this, we use atomic
        # min to ensure we give back the lowest index of an impostor
        cuda.atomic.min(answer, 0, upper)


# Arbitrarily-chosen block size, and a grid large enough to have one thread per
# window.
n_threads = 256
n_blocks = ceil(len(seq) / n_threads)

# Instead of returning the first impostor value, we instead return its index -
# this is because we can't use list.index (or np.where) later on in part 2 to
# get the index.
impostor_idx = np.asarray([len(seq) + 1], dtype=np.uint32)

# Launch the kernel
find_impostor[n_blocks, n_threads](seq, impostor_idx)

# Look up the actual value of the solution based on the index found by the
# kernel.
impostor = seq[impostor_idx[0]]
print(f"The answer to part 1 is {impostor}")


# Part 2

# The kernel for part 2 parallelises using one thread per starting index.
@cuda.jit
def find_contiguous(impostor_idx, seq, answer):
    i = cuda.grid(1)

    # Ensure we're not working on elements beyond the index of the impostor
    if i >= impostor_idx:
        return

    for j in range(impostor_idx - i):
        # We can't use sum, min, and max on arrays in the CUDA target, so we
        # implement them ourselves using a loop and an accumulator for each
        candidate_sum = 0
        # Starting minimum value chosen to be large enough that the initial
        # minimum value we find will be smaller than this
        candidate_min = 999999999999999
        candidate_max = 0
        for k in range(i, i+j):
            current = seq[k]
            candidate_sum += current
            candidate_min = min(candidate_min, current)
            candidate_max = max(candidate_max, current)

        if candidate_sum == impostor:
            # No atomic operation here as there's only one solution to be found
            answer[0] = candidate_min + candidate_max


# Allocate space for result
answer = np.zeros(1, dtype=np.uint32)

# Make the grid large enough for one thread per starting point, up to the
# location of the impostor
n_blocks = ceil(impostor_idx / n_threads)

# Kernel launch
find_contiguous[n_blocks, n_threads](impostor_idx[0], seq, answer)

print(f"The answer to part 2 is {answer[0]}")
