# SPDX-License-Identifier: BSD-3-Clause

import re

with open('input') as f:
    batch = f.read()

lines = [line.strip() for line in batch.split('\n\n')]

# Part 1

count = 0
for line in lines:
    fields = line.split()
    have_fields = {
        'byr': False,
        'iyr': False,
        'eyr': False,
        'hgt': False,
        'hcl': False,
        'ecl': False,
        'pid': False
    }

    for field in fields:
        if field[3] != ':':
            raise ValueError("Unexpected character in field")
        have_fields[field[:3]] = True

    valid = True
    for v in have_fields.values():
        valid = valid and v

    if valid:
        count += 1

print(f"The answer to part 1 is {count}")


# Part 2

count = 0
for line in lines:
    fields = line.split()
    valid_fields = {
        'byr': False,
        'iyr': False,
        'eyr': False,
        'hgt': False,
        'hcl': False,
        'ecl': False,
        'pid': False
    }

    for field in fields:
        key, value = field.split(':')
        if key == 'byr':
            valid_fields[key] = 1920 <= int(value) <= 2002
        if key == 'iyr':
            valid_fields[key] = 2010 <= int(value) <= 2020
        if key == 'eyr':
            valid_fields[key] = 2020 <= int(value) <= 2030
        if key == 'hgt':
            unit = value[-2:]
            valid_unit = False
            if unit == 'cm':
                lower, upper = 150, 193
                valid_unit = True
            elif unit == 'in':
                lower, upper = 59, 76
                valid_unit = True
            if valid_unit:
                valid_fields[key] = lower <= int(value[:-2]) <= upper
        if key == 'hcl':
            valid_start = value[0] == '#'
            rex = re.compile('[0-9a-f]{6}')
            valid_chars = rex.match(value[1:]) is not None
            valid_fields[key] = valid_start and valid_chars
        if key == 'ecl':
            valid_fields[key] = value in ('amb', 'blu', 'brn', 'gry', 'grn',
                                          'hzl', 'oth')
        if key == 'pid':
            correct_length = len(value) == 9
            rex = re.compile('[0-9]{9}')
            correct_value = rex.match(value) is not None
            valid_fields[key] = correct_length and correct_value

    valid = True
    for v in valid_fields.values():
        valid = valid and v

    if valid:
        count += 1

print(f"The answer to part 2 is {count}")
