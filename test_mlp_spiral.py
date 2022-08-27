"""
@file  test_mlp_spiral.py
@brief Test of sprial dataset using multiple layer perceptron.
"""

import numpy as np
import matplotlib.pyplot as plt
import nnet
from nnet.optim import SGD
from nnet.utils.data import Spiral
from nnet.utils.data import DataLoader
import nnet.nn.functional as F

class MLPSpiralNet(nnet.nn.Module):
    def __init__(self, dim_input, fc_output_sizes):
        super(MLPSpiralNet, self).__init__()

        layers = []
        for dim_output in fc_output_sizes[:-1]:
            layer = nnet.nn.Linear(dim_input, dim_output)
            dim_input = dim_output
            layers.append(layer)
            layers.append(nnet.ReLU())
        
        # outptu layer - no activation.
        layer = nnet.nn.Linear(dim_input, fc_output_sizes[-1])
        layers.append(layer)

        # lists shall be assigned after all the layers are appended to the list.
        self.layers = layers

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
            
        return y

    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return nnet.utils.plot_dot_graph(y, verbose=True, to_file=to_file)


def accuracy(y, y_pred):
    y = nnet.as_tensor(y)
    y_pred = nnet.as_tensor(y_pred)

    cls_pred = y_pred.data.argmax(axis=-1).reshape(y.shape)
    result = (cls_pred == y.data)
    acc = result.mean()
    return nnet.Tensor(acc)


batch_size = 30
spiral_train = Spiral(train=True)
train_loader = DataLoader(spiral_train, batch_size=batch_size, shuffle=True)

spiral_test = Spiral(train=False)

max_epochs = 300

model = MLPSpiralNet(2, (30, 3))
optimizer = SGD(lr=1.0)
optimizer.setup(model)

for epoch in range(max_epochs):
    sum_loss = 0
    sum_acc = 0

    for idx_batch, (x, y) in enumerate(train_loader):
       
        y_pred = model(x)
        loss = F.softmax_cross_entropy(y_pred, y)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        acc = accuracy(y, y_pred)
        sum_loss += float(loss.data) * len(y)
        sum_acc += float(acc.data) * len(y)

    avg_loss = sum_loss / len(spiral_train)
    avg_acc = sum_acc / len(spiral_train)
    print("epoch %d, loss %.2f, acc %.2f" % (epoch+1, avg_loss, avg_acc))


x = spiral_test.x
y = spiral_test.y

with nnet.no_grad():
    y_pred = model(x)

acc = accuracy(y, y_pred)
print("Accuracy: {}".format(acc))


x = x.numpy()
y = y.numpy()
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1

h = 0.001
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

with nnet.no_grad():
    y_pred = model(X)
y_pred = y_pred.numpy()

plt.figure('spiral')
y_pred = np.argmax(y_pred, axis=-1)
Z = y_pred.reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=3, colors=['#FFE1E9', '#97C1A9', '#9AB7D3', '#9AB7D3'])
marker = []
for yi in y:
    if yi == 0:
        marker.append('r')
    elif yi == 1:
        marker.append('g')
    else:
        marker.append('b')

plt.scatter(x[:,0], x[:,1], c=marker)
plt.savefig('images/spiral.png')
plt.show()

