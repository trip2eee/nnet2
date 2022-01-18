from nnet.optim.optimizer import Optimizer
import numpy as np

class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0):
        super(SGD, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        
        v -= self.lr * param.grad.data

        param.data += v

