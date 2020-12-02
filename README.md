Advent of Numba
===============

Solutions to [Advent of Code](https://adventofcode.com) using Numba. Some notes
on the solutions:

* Numba 0.52 is required to run the solutions.
  * I may use features that only appear in the master branch in later days.
* I will try to use CUDA for each solution.
* Most solutions will be the easiest for me to write.
  * This might mean a lot of brute force, due to the capabilities of a GPU and
    the low effort needed to invent brute force solutions.
* I will try to demonstrate something "interesting" about the CUDA target for
  each solution. E.g.:
  * Atomic operations
  * Cooperative grids
  * etc.
* I will try to annotate each solution to explain to a beginner the rationale
  behind the implementation.
* I will probably fall a few days behind.
* I am not optimistic about finishing all 24 days.

Please direct comments / questions / criticisms / veneration to:
[@gmarkall](https://twitter.com/gmarkall).
