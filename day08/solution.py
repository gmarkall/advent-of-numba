import numpy as np
from numba import njit

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


@njit
def execute(binary, visited):
    pc = 0
    accumulator = 0
    while not visited[pc]:
        visited[pc] = True
        instr = binary[pc]

        # Extract opcode
        opcode = (instr & 0xC000) >> 14

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
    return accumulator


visited = np.zeros(len(lines), dtype=np.bool_)
result = execute(binary, visited)
print(f'The answer to part 1 is {result}')
