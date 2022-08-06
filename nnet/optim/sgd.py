from nnet.optim.optimizer import Optimizer
import numpy as np
import nnet.cuda

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.
    """
    def __init__(self, lr=0.01, momentum=0.0):
        super(SGD, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        xp = nnet.cuda.get_array_module(param.data)

        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        
        v -= self.lr * param.grad.data

        param.data += v

