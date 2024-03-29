"""
@file  test_linear_layer.py
@brief Test of linear regression using multiple layer perceptron.
"""

import numpy as np
import matplotlib.pyplot as plt

import nnet
import nnet.nn
import nnet.nn.functional
import nnet.optim

np.random.seed(0)
x = np.random.rand(100, 1).astype(np.float32)
x.sort(axis=0)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1).astype(np.float32)

class RegressionNet(nnet.nn.Module):
    def __init__(self, fc_output_sizes):
        super(RegressionNet, self).__init__()

        layers = []

        dim_input = 1
        for dim_output in fc_output_sizes:
            layer = nnet.nn.Linear(dim_input, dim_output)
            dim_input = dim_output
            layers.append(layer)

        # lists shall be assigned after all the layers are appended to the list.
        self.layers = layers
        
    def forward(self, x):
        y = x
        for l in self.layers[:-1]:
            y = l(y)
            y = nnet.nn.functional.sigmoid(y)
        y = self.layers[-1](y)
        return y

    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return nnet.utils.plot_dot_graph(y, verbose=True, to_file=to_file)

model = RegressionNet((10, 1))

lr = 0.2
optimizer = nnet.optim.SGD(lr=lr, momentum=0.9)
optimizer.setup(model)

iters = 10000
for i in range(iters):
    y_pred = model(x)
    loss = nnet.nn.functional.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    print("{}: loss: {}".format(i, loss))

plt.figure('sin')
plt.scatter(x, y)
plt.plot(x, y_pred.data, c='r')
plt.savefig('images/linear_reg.png')
plt.show()


# 9999: loss: tensor(0.0745359468460083)
# 9999: loss: tensor(0.07226653575897217)