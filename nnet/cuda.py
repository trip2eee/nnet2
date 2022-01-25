import numpy as np
gpu_enable = True

try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False

from nnet import Tensor

def get_array_module(x):
    if isinstance(x, Tensor):
        x = x.data
    
    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp

def as_numpy(x):
    """This function converts cupy.ndarray into numpy.ndarray.    
    """
    if isinstance(x, Tensor):
        x = x.data
    
    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)
    
def as_cupy(x):
    if isinstance(x, Tensor):
        x = x.data
    
    if not gpu_enable:
        raise Exception('GPU is not enabled.')
    return cp.asarray(x)

    