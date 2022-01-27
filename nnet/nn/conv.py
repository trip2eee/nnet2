from nnet.nn.module import *
import numpy as np
import nnet.nn.functional as F

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dtype=np.float32):
        """Constructor of 2D convolution layer.
           in_channels (int) - Number of channels in the input image
           out_channels (int) - Number of channels produced by the convolution
           kernel_size (int or tuple) - Size of the convolving kernel
           stride (int or tuple) - Stride of the convolution. Default: 1
           padding (int or tuple) - Padding added to all four sides of the input. Default: 0
           bias (bool, optional) - If True, adds a learnable bias to the output. Default: True
        """
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        self._init_W()
        
        if bias:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')
        else:
            self.b = None
        
    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = nnet.utils.utils.pair(self.kernel_size)
        
        scale = np.sqrt(1.0 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data
    
    def forward(self, x):
        y = F.conv2d(x=x, W=self.W, b=self.b, stride=self.stride, pad=self.padding)
        return y




