# SPDX-License-Identifier: BSD-3-Clause

import itertools

USE_EXAMPLE = False

if USE_EXAMPLE:
    filename = 'example_input'
    preamble_length = 5
else:
    filename = 'input'
    preamble_length = 25

with open(filename) as f:
    seq = [int(x.strip()) for x in f.readlines()]


def find_impostor(seq):
    lower = 0
    upper = preamble_length
    while upper < len(seq):
        current = seq[upper]
        window = seq[lower:upper]

        found = False
        for x, y in itertools.product(window, window):
            if x == y:
                continue
            if (x + y) == current:
                found = True
                lower += 1
                upper += 1
                break

        if not found:
            return current


impostor = find_impostor(seq)
print(f"The answer to part 1 is {impostor}")


def find_contiguous(impostor, seq):
    loc = seq.index(impostor)
    for i in range(loc):
        for j in range(loc - i):
            candidate = seq[i:i+j]
            if sum(candidate) == impostor:
                return (min(candidate) + max(candidate))


answer = find_contiguous(impostor, seq)
print(f"The answer to part 2 is {answer}")
