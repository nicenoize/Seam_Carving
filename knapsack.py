import sys
import numpy as np

def knapsackId(items, maxweight):

    lookup = np.zeros((maxweight+1, len(items)+1))
    for j, (value, weight) in enumerate(items):
        for capacity in range(maxweight+1):
            if weight < capacity:
                val1 = lookup[capacity, j-1]
                val2 = lookup[capacity - weight, j-1]
                lookup[capacity, j] = max(val1, val2)
            else:
                lookup[capacity, j] = lookup[capacity, j-1]
    selection = []
    i = len(items)
    j = maxweight
    while i > 0:
        if lookup[j, i] != lookup[j, i-1]:
            selection.append(items[i-1])
            j -= items[i-1][1]
        i -= 1
    return lookup[maxweight, len(items)], selection

maxweight = 7
items = [[16,2],[19,3],[23,4],[28,5]]
value, selected_items = knapsackId(items, maxweight)

print("bag is worth:", value)
print("selected items:", selected_items)