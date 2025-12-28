import torch
import torch.nn as nn

x = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16)
conv = nn.Conv2d(3, 96, kernel_size=4, stride=4).cuda().half()

with torch.no_grad():
    y = conv(x)

print(y.shape)
