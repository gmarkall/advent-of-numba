import numpy as np
from math import ceil
from numba import cuda, njit, types

# Read in input

with open('input') as f:
    lines = f.readlines()

# Convert the input into an assembled binary for a little virtual machine.
# Each instruction encoded as a 16 bit word where the first two bits represent
# the opcode, and the remaining 14 bits are the operand in signed
# two's-complement representation.

NOP = 0x0
ACC = 0x1
JMP = 0x2

opcodes = {
    'nop': NOP,
    'acc': ACC,
    'jmp': JMP
}


# A little assembler for the language and VM

def assemble(lines):
    binary = np.zeros(len(lines), dtype=np.uint16)
    for i, line in enumerate(lines):
        mnemonic, operand = line.split()
        operand_value = np.uint16(int(operand))
        encoded = (opcodes[mnemonic] << 14) | (operand_value & 0x3FFF)
        binary[i] = encoded
    return(binary)


binary = assemble(lines)
n_instrs = len(lines)


# Part 1

# A nopython-mode function to solve part 1. Since part 1 is inherently
# sequential, it makes sense to implement as a nopython-mode function on the
# CPU rather than using CUDA. It is a little overcomplicated for part 1, but
# contains bits we need for its reuse in part 2 to demonstrate calling nopython
# mode functions from CUDA-jitted functions. The mechanisms only used in part 2
# are:
#
# Termination: for part 1 there is only an infinite loop, so the termination
# flag is never set.
# Mutation: We're just executing the binary as-is for part 1, so we don't
# mutate anything.

@njit
def execute(binary, visited, terminated, mutation):
    pc = 0
    accumulator = 0
    while not visited[pc]:
        visited[pc] = True
        instr = binary[pc]

        # Extract opcode
        opcode = (instr & 0xC000) >> 14

        # Mutate the opcode if required
        if pc == mutation:
            if opcode == NOP:
                opcode = JMP
            elif opcode == JMP:
                opcode = NOP

        # Decode signed 14-bit operand
        operand = (instr & 0x3FFF)
        if (operand & 0x2000):
            operand = - (16384 - operand)

        # Execute instruction
        if opcode == NOP:
            pc += 1
        elif opcode == ACC:
            accumulator += operand
            pc += 1
        elif opcode == JMP:
            pc += operand

        if pc == n_instrs:
            terminated[0] = True
            return accumulator

    return accumulator


# For keeping track of which instructions we've visited
visited = np.zeros(n_instrs, dtype=np.bool_)

# Don't mutate any instruction for part 1 - using -1 as the PC of the
# instruction to mutate ensures none match.
mutation = -1
# We don't care about termination for part 1, but need to pass something in.
terminated = np.zeros(1, dtype=np.bool_)

# Execute the binary until an infinite loop is hit.
result = execute(binary, visited, terminated, mutation)
print(f'The answer to part 1 is {result}')


# Part 2 - reuse the executor from part 1, using one thread to try a mutation
# of each instruction. This demonstrates how the kernel of a function can be
# shared between CPU and CUDA targets, with the CUDA-specific portion
# orchestrating the allocation of work to threads.

@cuda.jit
def brute_force_mutations(binary, visited, answer):
    i = cuda.grid(1)
    if i >= n_instrs:
        return

    # Each thread needs its own termination flag
    terminated = cuda.local.array(1, dtype=types.boolean)
    # Call njit function
    result = execute(binary, visited[i], terminated, i)

    # Did our mutated variant terminate? If so, set the result.
    if terminated[0]:
        answer[0] = result


# We need a 2D array of visited instructions this time - one entry per
# instruction per mutation.
visited = np.zeros((n_instrs, n_instrs), dtype=np.bool_)
# Allocate space for the result
answer = np.zeros(1, dtype=np.uint32)

n_threads = 256
n_blocks = ceil(n_instrs / n_threads)

brute_force_mutations[n_blocks, n_threads](binary, visited, answer)
print(f'The answer to part 2 is {answer[0]}')
