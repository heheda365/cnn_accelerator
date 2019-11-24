
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn as nn
import numpy as np

class Shift3x3(nn.Module):
    '''
    用移动数据方式计算 3 * 3卷积
    '''
    def __init__(self, planes):
        super(Shift3x3, self).__init__()

        self.planes = planes
        kernel = np.zeros((planes, 1, 3, 3), dtype=np.float32)

        for i in range(planes):
            if i % 5 == 0:
                kernel[i, 0, 0, 1] = 1.0
            elif i % 5 == 1:
                kernel[i, 0, 1, 0] = 1.0
            elif i % 5 == 2:
                kernel[i, 0, 1, 1] = 1.0
            elif i % 5 == 3:
                kernel[i, 0, 1, 2] = 1.0
            else:
                kernel[i, 0, 2, 1] = 1.0

        self.register_parameter('bias', None)
        self.kernel = nn.Parameter(
            torch.from_numpy(kernel), requires_grad=False)

    def forward(self, input):
        return F.conv2d(input,
                        self.kernel,
                        self.bias,
                        (1, 1),  # stride
                        (1, 1),  # padding
                        1,  # dilation
                        self.planes,  # groups
                        )