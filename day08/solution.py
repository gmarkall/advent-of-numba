import numpy as np
from math import ceil
from numba import cuda, njit, types

with open('input') as f:
    lines = f.readlines()

NOP = 0x0
ACC = 0x1
JMP = 0x2

opcodes = {
    'nop': NOP,
    'acc': ACC,
    'jmp': JMP
}


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
            #print(f"Mutating {pc}")
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


visited = np.zeros(n_instrs, dtype=np.bool_)

# Don't mutate any instruction
mutation = -1
terminated = np.zeros(1, dtype=np.bool_)
result = execute(binary, visited, terminated, mutation)
print(f'The answer to part 1 is {result}')


@cuda.jit
def brute_force_mutations(binary, visited, answer):
    i = cuda.grid(1)
    if i >= n_instrs:
        return

    terminated = cuda.local.array(1, dtype=types.boolean)
    result = execute(binary, visited[i], terminated, i)

    if terminated[0]:
        #print(f"Thread {i} finds answer {result}")
        answer[0] = result


visited = np.zeros((n_instrs, n_instrs), dtype=np.bool_)
answer = np.zeros(1, dtype=np.uint32)

n_threads = 256
n_blocks = ceil(n_instrs / n_threads)

brute_force_mutations[n_blocks, n_threads](binary, visited, answer)
print(f'The answer to part 2 is {answer[0]}')
