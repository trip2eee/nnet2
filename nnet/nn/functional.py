from typing_extensions import final

from numpy.lib.stride_tricks import broadcast_to
from nnet.function import Function
from nnet.tensor import Tensor
import numpy as np
import nnet.utils.utils

def as_array(x):
    if np.isscalar(x):
        y = np.array(x)
    else:
        y = x
    return y

def as_tensor(obj):
    if isinstance(obj, Tensor):
        return obj
    return Tensor(obj)

class Add(Function):
    def forward(self, x0, x1):
        """y = x0 + x1
        """
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 + x1
        return y
    
    def backward(self, gy):
        """dy/dx0 = 1
           dy/dx1 = 1
        """
        gx0 = gy
        gx1 = gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)            
        return gx0, gx1

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

class Sub(Function):
    """Subtraction
    """
    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 - x1
        return y
    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

class Mul(Function):
    def forward(self, x0, x1):
        """y = x0 * x1
        """
        y = x0 * x1
        return y
    
    def backward(self, gy):
        """dy/dx0 = x1
           dy/dx1 = x0
        """
        x0 = self.inputs[0]
        x1 = self.inputs[1]
        gx0 = gy*x1
        gx1 = gy*x0

        if x0.shape != x1.shape:
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1

def mul(x0, x1):
    return Mul()(x0, x1)

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    def backward(self, gy):
        """y = x0/x1
           dy/dx0 = 1/x1
           dy/dx1 = -x0/x1^2
        """
        x0 = self.inputs[0]
        x1 = self.inputs[1]        
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)

        if x0.shape != x1.shape:
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

class Square(Function):
    def forward(self, x):
        return x**2
    
    def backward(self, gy):
        x = self.inputs[0]
        return 2 * x * gy

def square(x):
    return Square()(x)

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0]
        gx = np.exp(x) * gy
        return gx

def exp(x):
    return Exp()(x)

class Neg(Function):
    """Negation
    """
    def forward(self, x):
        return -x
    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)

class Pow(Function):
    def __init__(self, c):
        super(Pow,self).__init__()
        self.c = c
    
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        """y=x^c
           dy/dx = c*x^(c-1)
        """
        x = self.inputs[0]
        c = self.c
        gx = c* x ** (c - 1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x = self.inputs[0]
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    def backward(self, gy):
        x = self.inputs[0]
        gx = gy * -np.sin(x)
        return gx

def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - (y * y))
        return gx

def tanh(x):
    return Tanh()(x)

def reshape(x, shape):
    if x.shape == shape:
        return as_tensor(x)
    return Reshape(shape)(x)

class Reshape(Function):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)

def transpose(x):
    return Transpose()(x)

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    
    def backward(self, gy):
        gx = transpose(gy)
        return gx

class Sum(Function):
    def __init__(self, axis, keepdims):
        super(Sum, self).__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = nnet.utils.utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        super(BroadcastTo, self).__init__()
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_tensor(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        super(SumTo, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = nnet.utils.utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_tensor(x)
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def matmul(x, W):
    return MatMul()(x, W)

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)

class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

class Sigmoid(Function):
    def forward(self, x):
        # y = 1 / (1 + exp(-x))
        y = np.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)

class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx

def relu(x):
    return ReLU()(x)

class GetItem(Function):
    def __init__(self, slices):
        super(GetItem, self).__init__()
        self.slices = slices
    
    def forward(self, x):
        y = x[self.slices]
        return y
    
    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)
    
def get_item(x, slices):
    return GetItem(slices)(x)

class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        super(GetItemGrad, self).__init__()
        self.slices = slices
        self.in_shape = in_shape
    
    def forward(self, gy):
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx
    
    def backward(self, ggx):
        return get_item(ggx, self.slices)

class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

def softmax(x, axis=1):
    return Softmax(axis)(x)

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = nnet.utils.utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)