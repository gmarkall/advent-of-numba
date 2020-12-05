# SPDX-License-Identifier: BSD-3-Clause

from math import ceil
from numba import cuda, types
import numpy as np


# The idea here was to read in the input as a sequence of bytes and do all
# processing on the GPU. Unfortunately this didn't work out in the time I had
# available, but I leave my incomplete solution here in case I return to it, or
# in case it is of interest.

with open('example_input') as f:
    encoded = f.read().encode('utf-8')
    np_batch = np.frombuffer(encoded, dtype=np.uint8)
    batch = cuda.to_device(np_batch)


print(batch)

n_bytes = len(batch)
newline_char = ord('\n')

cb = ord('b')
cy = ord('y')
cr = ord('r')
ci = ord('i')
ce = ord('e')
ch = ord('h')
cl = ord('l')
cp = ord('p')
ccolon = ord(':')


@cuda.jit
def validate_passports(batch, newline_counter, newline_locs, answer):
    # Thread ID, as usual
    i = cuda.grid(1)
    # Grid group - used for synchronizing all threads within the grid
    grid = cuda.cg.this_grid()

    # Find the locations of the blank lines. One thread per byte of input
    # (looking at the adjacent byte as well), each checks if it is on a blank
    # line. Since we don't know which threads will find blank lines, we
    # atomically increment a count of position in the array storing the
    # positions of blank lines.
    if i < (n_bytes - 1):
        if batch[i] == newline_char and batch[i + 1] == newline_char:
            nli = cuda.atomic.inc(newline_counter, 0, n_bytes)
            newline_locs[nli] = i

    # Now we know where all the blank lines are, but not in order.

    # Do a grid sync so that every thread can see all the blank line locations.
    grid.sync()

    # Debugging - was using this to see where the kernel got to
    if i == 0:
        print(9901)


    # Sort the list of blank lines. Commented code is a whole-grid odd-even
    # parallel sort I started writing (which works to some extent), but I
    # switched to using a single-thread version to make life easier whilst
    # developing. See e.g. Chapter 46 from GPU Gems2:
    # https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting

#    my_index = i * 2
#    even = False
#    newline_count = newline_counter[0]
#
#    if i == 0:
#        print(9901)
#
#    if my_index < (newline_count - 2):
#        while True:
#            if i == 0:
#                print(9902)
#            # Initialise the count of changes
#            if i == 0:
#                answer[0] = 0
#            # Wait until every thread can see that the count is 0
#            grid.sync()
#
#            if i == 0:
#                print(9903)
#
#            # Determine indices into the array being sorted based on whether
#            # we're in an odd or even pass
#            ri1 = my_index
#            ri2 = my_index + 1
#            if even:
#                ri1 += 1
#                ri2 += 1
#
#            # Read in the values for comparison
#            p1, p2 = newline_locs[ri1], newline_locs[ri2]
#
#            # Swap the values if necessary
#            if p1 < p2:
#                newline_locs[ri1] = p1
#                newline_locs[ri2] = p2
#            else:
#                newline_locs[ri1] = p2
#                newline_locs[ri2] = p1
#                # If we swapped something, make a note of that
#                cuda.atomic.inc(answer, 0, n_bytes)
#
#            if i == 0:
#                print(9904)
#
#            # Wait until every thread has finished this pass
#            grid.sync()
#
#            if i == 0:
#                print(9905)
#
#            if i == 0:
#                print(answer[0])
#            # If no thread swapped its pair, the list is now in order
#            if answer[0] == 0:
#                break
#
#            # Swap between odd and even passes for the next iteration
#            even = not even

    # Quick implementation of single-threaded sort. Parallelise later.
    if i == 0:
        changed = True
        while changed:
            changed = False
            for i in range(newline_counter[0] - 1):
                p1, p2 = newline_locs[i], newline_locs[i + 1]
                if p2 < p1:
                    newline_locs[i] = p2
                    newline_locs[i + 1] = p1
                    changed = True
        
        print(99021)

    #print(990000 + i)
    if i == 0:
        print(9989)

    # Now we've sorted the list, sync the whole grid again to make sure all
    # threads can see the sorted list.
    grid.sync()

    if i == 0:
        print(9902)


    # Very incomplete for the rest of the kernel - the idea was for one thread
    # per line to check each line for the required fields (then parallelise
    # further by one thread per byte if I got this working).

    if i < newline_counter[0] + 1:
        if i == 0:
            lb = 0
        else:
            lb = newline_locs[i - 1]
        ub = newline_locs[i]

        # Assume no line greater than 128 chars
        my_line = cuda.local.array(128, dtype=types.uint8)

        have_byr = False
        have_iyr = False
        have_eyr = False
        have_hgt = False
        have_hcl = False
        have_ecl = False
        have_pid = False

        for p in range((ub - lb) - 4):
            chars = my_line[p:p+4]
            if i == 3:
                print(chars)
            if chars[3] != ccolon:
                continue
            if chars[0] == cb and chars[1] == cy and chars[2] == cr:
                have_byr = True

        valid = have_byr

    if i == 0:
        cuda.atomic.exch(answer, 0, 0)

    if i == 0:
        print(9904)

    grid.sync()

    if i == 0:
        print(9905)

    #if valid:
    #    cuda.atomic.inc(answer, 0, newline_counter[0])


n_threads = 256
n_blocks = ceil(n_bytes / n_threads)

answer = np.zeros(1, dtype=np.uint32)
newline_counter = np.zeros(1, dtype=np.uint32)
newline_locs = cuda.device_array(n_bytes, dtype=np.uint32)
validate_passports[n_blocks, n_threads](batch, newline_counter, newline_locs,
                                        answer)

cuda.synchronize()

print(newline_locs.copy_to_host())
print(answer)
