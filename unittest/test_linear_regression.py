import unittest
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
current_dir = os.path.realpath(os.path.curdir)
sys.path.insert(0, current_dir)
import nnet

class TestLinearRegression(unittest.TestCase):
    def test_linear_regression(self):
        np.random.seed(0)

        x = np.random.rand(100, 1)
        y = 5 + 2 * x + np.random.rand(100, 1)
        x = nnet.Tensor(x)
        y = nnet.Tensor(y)

        W = nnet.Tensor(np.zeros((1, 1)))
        b = nnet.Tensor(np.zeros(1))

        def predict(x):
            y = nnet.matmul(x, W) + b
            return y
        
        lr = 0.1
        iters = 100

        for i in range(iters):
            y_pred = predict(x)
            loss = nnet.mean_squared_error(y, y_pred)

            W.cleargrad()
            b.cleargrad()
            loss.backward()

            W.data -= lr * W.grad.data
            b.data -= lr * b.grad.data

            print(W, b, loss)            
        
        y_pred = predict(x)
        loss = nnet.mean_squared_error(y, y_pred)
        self.assertLess(loss.data, 0.0791)

        plt.figure('regression')
        plt.scatter(x.data, y.data)
        x = np.array([x.data.min(), x.data.max()])
        y = W.data[0,0]*x + b.data[0]
        plt.plot(x, y, c='r')
        plt.show()

if __name__ == "__main__":
    unittest.main()