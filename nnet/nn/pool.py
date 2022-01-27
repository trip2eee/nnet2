from nnet.nn.module import *
import nnet.nn.functional as F


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        """Constructor of 2D convolution layer.
           kernel_size (int or tuple) - The size of the window to take a max over.
           stride (int or tuple) - The stride of the window. Default: kernel_size
           padding (int or tuple) - Padding added to all four sides of the input. Default: 0
        """
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size

        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        y = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        return y




