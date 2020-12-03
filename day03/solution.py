from math import ceil
from numba import cuda
import numpy as np

with open('input') as f:
    lines = [line.strip() for line in f.readlines()]

shape = len(lines), len(lines[0])

slope = np.zeros(shape, dtype=np.bool_)

for i, line in enumerate(lines):
    for j, c in enumerate(line):
        slope[i, j] = (c == '#')

print(slope)


@cuda.jit
def count_trees(slope, answer):
    i = cuda.grid(1)
    if i < shape[0]:
        j = (i * 3) % shape[1]
        if slope[i, j]:
            cuda.atomic.inc(answer, 0, shape[0])


n_threads = 256
n_blocks = ceil(shape[0] / n_threads)

answer = np.zeros(1, dtype=np.uint32)
count_trees[n_blocks, n_threads](slope, answer)
print(answer)


@cuda.jit
def count_trees_parameterised(slope, right, down, answer, idx):
    tid = cuda.grid(1)
    i = tid * down
    if i < shape[0]:
        j = (tid * right) % shape[1]
        if slope[i, j]:
            cuda.atomic.inc(answer, idx, shape[0])


partials = cuda.to_device(np.zeros(5, dtype=np.uint32))
for i, (right, down) in enumerate(((1, 1), (3, 1), (5, 1), (7, 1), (1, 2))):
    count_trees_parameterised[n_blocks, n_threads](slope, right,
                                                   down, partials, i)


@cuda.reduce
def mul_reduce(x, y):
    return x * y


print(partials.copy_to_host())
answer = mul_reduce(partials, init=1)
print(answer)
