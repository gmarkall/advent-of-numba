# SPDX-License-Identifier: BSD-3-Clause

from math import ceil
from numba import cuda
import numpy as np

# Read in input - create a 2D boolean array for the slope, where True
# represents a tree.

with open('input') as f:
    lines = [line.strip() for line in f.readlines()]

shape = len(lines), len(lines[0])

slope = np.zeros(shape, dtype=np.bool_)

for i, line in enumerate(lines):
    for j, c in enumerate(line):
        slope[i, j] = (c == '#')


# Part 1 - count the trees using one thread per input line

@cuda.jit
def count_trees(slope, answer):
    i = cuda.grid(1)
    if i < shape[0]:
        # Modulo used to implement the repeating pattern of the slope
        j = (i * 3) % shape[1]
        if slope[i, j]:
            cuda.atomic.inc(answer, 0, shape[0])


# Choose a reasonable block size and make the grid big enough
n_threads = 256
n_blocks = ceil(shape[0] / n_threads)


# Allocate space for result, and launch kernel
answer = np.zeros(1, dtype=np.uint32)
count_trees[n_blocks, n_threads](slope, answer)

print(f"The answer to part 1 is {answer[0]}")


# Part 2 - use a parameterised version of the kernel from part 1.

@cuda.jit
def count_trees_parameterised(slope, right, down, answer, idx):
    tid = cuda.grid(1)
    i = tid * down
    if i < shape[0]:
        j = (tid * right) % shape[1]
        if slope[i, j]:
            cuda.atomic.inc(answer, idx, shape[0])


# We will instantiate the kernel once for each trajectory, and store the number
# of trees encountered in each trajectory in an array per trajectory.

# The number of trees for each trajectory is kept on the device rather than the
# host, so that we don't need to transfer data between the CPU and GPU between
# the calls to our tree counting kernel and the reduction kernel.
partials = cuda.to_device(np.zeros(5, dtype=np.uint32))

# We'll also move the slope data to the device so that it doesn't need a CPU ->
# GPU transfer for each launch.
device_slope = cuda.to_device(slope)

# (right, down) values listed in problem specification.
for i, (right, down) in enumerate(((1, 1), (3, 1), (5, 1), (7, 1), (1, 2))):
    count_trees_parameterised[n_blocks, n_threads](device_slope, right,
                                                   down, partials, i)


# Elementwise reduction function - this should take two inputs and return a
# single reduced output.
# See https://numba.readthedocs.io/en/stable/cuda/reduction.html
@cuda.reduce
def mul_reduce(x, y):
    return x * y


# Call the reduction kernel. The initial value for the reduction defaults to 0,
# so we need to specify an initial value of 1 for multiplication to produce the
# correct end result (otherwise we start with 0 * partials[0] and end up with
# 0).
#
# The result of the reduction is returned from the reduction kernel.
answer = mul_reduce(partials, init=1)

print(f"The answer to part 2 is {answer}")
