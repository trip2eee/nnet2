from nnet.nn.module import *
import nnet.nn.functional
import numpy as np

class Linear(Module):
    """Applies a linear transform to the incoming data: y = x*W+b
    """
    def __init__(self, in_features, out_features, bias=True, dtype=np.float32):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        
        w = np.random.randn(in_features, out_features).astype(dtype) * np.sqrt(1 / in_features)
        self.W = Parameter(w, name='W')

        if bias:
            b = np.zeros(out_features, dtype=dtype)
            self.b = Parameter(b, name='b')
        else:
            self.b = None

    def forward(self, x):
        y = nnet.nn.functional.linear(x, self.W, self.b)
        return y