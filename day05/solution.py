# SPDX-License-Identifier: BSD-3-Clause

from math import ceil
from numba import cuda
import numpy as np

# Read in input - create a 2D boolean array for the boarding passes
# False represents front / left, True is back / right.

with open('input') as f:
    lines = [line.strip() for line in f.readlines()]

shape = len(lines), len(lines[0])

passes = np.zeros(shape, dtype=np.bool_)

for i, line in enumerate(lines):
    for j, c in enumerate(line):
        passes[i, j] = (c == 'B') or (c == 'R')


# A device function - this can be called from a kernel (or another device
# function) and is used by the kernels for both parts.

@cuda.jit(device=True)
def compute_seat_id(boarding_pass):
    row = 0
    for j in range(7):
        add = 2 ** (6 - j)
        if boarding_pass[j]:
            row += add

    col = 0
    for j in range(7, 10):
        add = 2 ** (9 - j)
        if boarding_pass[j]:
            col += add

    return row * 8 + col


# Part 1 - use one thread per pass, and atomic maximum to find the single
# highest pass ID.
@cuda.jit
def highest_boarding_pass(passes, answer):
    i = cuda.grid(1)
    if i < shape[0]:
        seat_id = compute_seat_id(passes[i])
        cuda.atomic.max(answer, 0, seat_id)


# Choose a reasonable block size and make the grid big enough
n_threads = 256
n_blocks = ceil(shape[0] / n_threads)

# Allocate space for result, and launch kernel
answer = np.zeros(1, dtype=np.uint32)
highest_boarding_pass[n_blocks, n_threads](passes, answer)

print(f"The answer to part 1 is {answer[0]}")


# Part 2 - Mark all the filled seats in a boolean array. When we are finished
# marking, do a whole-grid sync so that all threads in the grid see the
# filled-in array. After that, each active thread checks one seat.
@cuda.jit
def find_my_seat(passes, filled_seats, answer):
    i = cuda.grid(1)

    # Fill out array of occupied seats
    if i < shape[0]:
        seat_id = compute_seat_id(passes[i])
        filled_seats[seat_id] = True

    # Writes to the seat array will not be visible to all threads until a grid
    # sync is done.
    grid = cuda.cg.this_grid()
    grid.sync()

    # Seat will not be the very first or the very last
    if (i > 0) and (i < (len(filled_seats) - 1)):
        # Check the seat - looking for an empty seat with adjacent seats
        # occupied.
        if filled_seats[i - 1] and filled_seats[i + 1] and not filled_seats[i]:
            # No atomic operation because there should only be one answer
            answer[0] = i


# Allocate a boolean array for filled seats, all False to begin with.
filled_seats = np.zeros(answer[0], dtype=np.bool_)

# Allocate space for result, and launch kernel
answer = np.zeros(1, dtype=np.uint32)
find_my_seat[n_blocks, n_threads](passes, filled_seats, answer)

print(f"The answer to part 2 is {answer[0]}")
