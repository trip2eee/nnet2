from nnet.nn.module import *

class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        """start_dim - first dim to flatten (default = 1).
           end_dim - last dim to flatten (default = -1).
        """
        super(Flatten, self).__init__()
        self.x_shape = None
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x):
        if self.x_shape is None:
            self.x_shape = list(x.shape)
            self.y_shape = self.x_shape[0:self.start_dim] + [-1]
            if self.end_dim != -1:
                self.y_shape = self.y_shape + self.x_shape[self.end_dim+1:]
        
        y = x.reshape(self.y_shape)
        return y

    def backward(self, gy):
        y = gy.reshape(self.x_shape)
        return y

