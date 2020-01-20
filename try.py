import torch
from torch.nn.functional import conv2d
x = torch.ones((1,512, 7, 6))
x[0, 0, 0, 0] = 0
print(x)
kernel = torch.ones((3,3))
mid = 1
kernel[mid, mid] = -8
kernel = kernel/-8
print(kernel)

b, c, h, w = x.shape
kernel = kernel.type(torch.FloatTensor)
kernel = kernel.repeat(c, 1, 1, 1)
ans = conv2d(x, kernel, stride=1, groups=c)
print(ans)
print(torch.sum(ans))
