import nnet
import nnet.nn
import nnet.utils.data
import nnet.optim
import matplotlib.pyplot as plt
import nnet.nn.functional as F
import numpy as np
import time
import cProfile

np.random.seed(123)

batch_size = 500
max_epochs = 5

train_set = nnet.utils.data.MNIST(train=True, transform=None)
test_set = nnet.utils.data.MNIST(train=False, transform=None)

train_loader = nnet.utils.data.DataLoader(train_set, batch_size, shuffle=True)
test_loader = nnet.utils.data.DataLoader(test_set, batch_size, shuffle=True)


class MLPNet(nnet.nn.Module):
    def __init__(self, dim_input, fc_output_sizes):
        super(MLPNet, self).__init__()

        layers = []
        layers.append(nnet.nn.Flatten())

        for dim_output in fc_output_sizes[:-1]:
            layer = nnet.nn.Linear(dim_input, dim_output)
            dim_input = dim_output
            layers.append(layer)
            layers.append(nnet.ReLU())
            layers.append(nnet.nn.Dropout(0.2))
        
        # outptu layer - no activation.
        layer = nnet.nn.Linear(dim_input, fc_output_sizes[-1])
        layers.append(layer)

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

device = 'gpu'
# device = 'cpu'

model = MLPNet(28*28, (1000, 1000, 10))
model.to(device)

optimizer = nnet.optim.Adam(lr=0.001)
optimizer.setup(model)


def train():
    for epoch in range(max_epochs):
        sum_loss = 0
        sum_acc = 0
        t_start = time.time()
        model.train()
        for idx_batch, (x, y) in enumerate(train_loader):

            x.to(device)
            y.to(device)

            y_pred = model(x)
            loss = F.softmax_cross_entropy(y_pred, y)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            
            acc = accuracy(y, y_pred)
            sum_loss += float(loss.data) * len(y)
            sum_acc += float(acc.data) * len(y)

        t_end = time.time()
        t_duration = (t_end - t_start)

        avg_loss = sum_loss / len(train_set)
        avg_acc = sum_acc / len(train_set)
        print("epoch {} - {:.2f} sec".format(epoch, t_duration))
        print("    train loss: {:.4f}, accuracy: {:.4f}".format(avg_loss, avg_acc))

        # test
        model.eval()
        sum_loss = 0
        sum_acc = 0
        with nnet.no_grad():
            for idx_batch, (x, y) in enumerate(test_loader):

                x.to(device)
                y.to(device)

                y_pred = model(x)
                loss = F.softmax_cross_entropy(y_pred, y)
                            
                acc = accuracy(y, y_pred)
                sum_loss += float(loss.data) * len(y)
                sum_acc += float(acc.data) * len(y)

            avg_loss = sum_loss / len(test_set)
            avg_acc = sum_acc / len(test_set)
            print("    test loss: {:.4f}, accuracy: {:.4f}".format(avg_loss, avg_acc))

    model.save_weights('mnist_model.npz')

def test():
    model.load_weights('mnist_model.npz')
    model.to(device)

    sum_loss = 0
    sum_acc = 0
    model.eval()
    with nnet.no_grad():
        for idx_batch, (x, y) in enumerate(test_loader):

            x.to(device)
            y.to(device)

            y_pred = model(x)
            loss = F.softmax_cross_entropy(y_pred, y)
                        
            acc = accuracy(y, y_pred)
            sum_loss += float(loss.data) * len(y)
            sum_acc += float(acc.data) * len(y)

        avg_loss = sum_loss / len(test_set)
        avg_acc = sum_acc / len(test_set)
        print("    test loss: {:.4f}, accuracy: {:.4f}".format(avg_loss, avg_acc))

if __name__ == "__main__":
    # cProfile.run('train()')
    train()
    test()