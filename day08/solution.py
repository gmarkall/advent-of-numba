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
for word in binary:
    print("%X" % word)

print()

@njit
def execute(binary, visited):
    pc = 0
    accumulator = 0
    while not visited[pc]:
        print(pc)
        visited[pc] = True
        instr = binary[pc]
        opcode = (instr & 0xC000) >> 14
        operand = (instr & 0x3FFF)
        if (operand & 0x2000):
            # SIgn extend to 16 bits if necessary
            operand = operand | 0xF000
        #print(operand)
        if (operand & 0x8000):
            operand = - (65536 - operand)
        #print(operand)
        #pc += 1
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
print()
print(result)