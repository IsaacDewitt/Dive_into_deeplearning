import numpy as np
import torch
from torch import nn
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utility import Timer

timer = Timer()

A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)

timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()

timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
timer.start()
A = torch.mm(B, C)
timer.stop()

 # 乘法和加法作为单独的操作（在实践中融合）
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')