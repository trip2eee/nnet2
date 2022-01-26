from nnet.nn.module import *

class Dropout(Module):
    def __init__(self, p=0.5):
        """p (float)) probability of an element to be zeroed. Default: 0.5
        """
        super(Dropout, self).__init__()
        self.p = p
    
    def forward(self, x):
        return nnet.nn.functional.dropout(x, self.p, self.train_mode)


