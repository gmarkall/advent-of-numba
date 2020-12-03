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
  * The solutions will probably not be close to performance-optimal.
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


Solutions
---------

Links to solutions and some interesting features of them:

* [Day 1](day01/solution.py): 2D / 3D grids, atomic exchange for stores.
* [Day 2](day02/solution.py): Atomic increment, structured arrays.
* [Day 3](day03/solution.py): Building reduction kernels with `@cuda.reduce`,
  host to device transfers to elide unnecessary copying


Other approaches
----------------

* [Jacob Tomlinson](https://jacobtomlinson.dev/) is also [using Numba CUDA to
  solve AoC](https://github.com/jacobtomlinson/advent-of-gpu-code-2020). He is
  also streaming and recording his work:
  * [YouTube](https://www.youtube.com/channel/UCjwcSpcyRYsfZMsliAJzYuQ)
  * [Twitch](https://www.twitch.tv/constrainedcoding)
