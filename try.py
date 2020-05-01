import torch
from torch.nn.functional import conv2d
x = torch.ones((1,1, 112, 96))
x[0, 0, 1, 0] = 0
x[0, 0, 1, 2] = 0

print(x)
kernel = torch.ones((3,3))
mid = 1
kernel[mid, mid] = -8
kernel = kernel/-8
print(kernel)
b, c, h, w = x.shape
print(c)
# c = 1
kernel = kernel.type(torch.FloatTensor)
kernel = kernel.repeat(c, 1, 1, 1)
ans = conv2d(x, kernel, stride=1, groups=c)
print(ans)
print(torch.sum(ans))
