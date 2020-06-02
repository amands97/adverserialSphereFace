import torch
import random

def get_random_indices(level, nrows = 7, ncols = 6):
    list_nums = []
    # for i in range(4*level):
    i = 0
    if 4 * level > nrows * ncols:
        print("error level > nrows * ncols")
        # import sys
        return None 
    while(i < 4* level):
        x = random.randint(0, nrows - 1)
        y = random.randint(0, ncols - 1)
        if tuple([x,y]) not in list_nums:
            list_nums.append(tuple([x,y]))
            i += 1
    list_nums = [list(index) for index in list_nums]
    list_nums = torch.LongTensor(list_nums)
    
    return list_nums

level = 10
output = torch.ones((2, 1, 7, 6))
for i, out in enumerate(output):
    indices = get_random_indices(level)
    mask = torch.ones((1, 7, 6))
    mask[0, indices[:, 0], indices[:, 1]] = 0

    output[i] = ((output[i] * mask))
print(output)