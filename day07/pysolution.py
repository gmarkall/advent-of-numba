# SPDX-License-Identifier: BSD-3-Clause

with open('input') as f:
    lines = [line.strip('.\n') for line in f.readlines()]

rules = {}

for line in lines:
    bag, contains = line.split(' bags contain ')
    contents = contains.split(', ')
    contents_list = []
    for content in contents:
        if content == 'no other bags':
            break
        parts = content.split()
        count = int(parts[0])
        description = " ".join(parts[1:-1])
        contents_list.append((count, description))
    rules[bag] = contents_list


def contains_shiny_gold(bag):
    contents = rules[bag]
    for content in contents:
        contained_bag = content[1]
        if contained_bag == 'shiny gold':
            return True
        if contains_shiny_gold(contained_bag):
            return True
    return False


total = 0

for bag in rules.keys():
    if contains_shiny_gold(bag):
        total += 1

print(f'The answer to part 1 is {total}')


def contains_count(bag):
    contents = rules[bag]
    total = 0
    for content in contents:
        count, contained_bag = content
        contained_count = contains_count(contained_bag)
        # Add how many bags the contained bags contain
        total += count * contained_count
        # Add the contained bags themselves
        total += count
    return total


print(f"The answer to part 2 is {contains_count('shiny gold')}")
