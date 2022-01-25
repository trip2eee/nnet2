from nnet.optim.optimizer import Optimizer
import numpy as np
import math
import nnet.cuda

class Adam(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        """lr - learning rate (default: 1e-3)
           betas - coefficients used for computing running average of gradient and its square (default:(0.9, 0.999))
           eps - tem added to the denominator to improve numerical stability (default: 1e-8)
        """
        super().__init__()
        self.t = 0
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    def update_one(self, param):
        xp = nnet.cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)    # m_0
            self.vs[key] = xp.zeros_like(param.data)    # v_0

        m = self.ms[key]    # m_t-1
        v = self.vs[key]    # v_t-1

        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps

        grad = param.grad.data

        m += (1 - beta1) * (grad - m)    
        v += (1 - beta2) * (grad * grad - v)

        """
        param = param - lr * mt^/(sqrt(vt^) + eps)
        \hat{mt} = mt / (1-beta1^t)
        \hat{vt} = vt / (1-beta2^2)
        \hat{mt} / (sqrt(\hat{vt}) + eps) = mt*(1-beta2^2) / vt*(1-beta1^t)
        """
        hat_m = m / (1.0 - math.pow(self.beta1, self.t))
        hat_v = v / (1.0 - math.pow(self.beta2, self.t))

        param.data -= self.lr * hat_m / (xp.sqrt(hat_v) + eps)