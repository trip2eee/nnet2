import unittest
import nnet
import nnet.nn.functional
import numpy as np

def numerical_diff(f, x, eps=1e-4):
    x0 = nnet.Tensor(x.data - eps)
    x1 = nnet.Tensor(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = nnet.Tensor(nnet.as_array(2.0))
        y = nnet.square(x)
        expected = np.array(4.0)

        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = nnet.Tensor(np.array(3.0))
        y = nnet.square(x)

        y.backward()
        expected = np.array(6.0)
        
        self.assertEqual(x.grad.data, expected)

    def test_gradient_check(self):
        x = nnet.Tensor(np.random.rand(1))

        y = nnet.square(x)
        y.backward()
        num_grad = numerical_diff(nnet.square, x)
        flg = np.allclose(x.grad.data, num_grad)
        self.assertTrue(flg)
    
    def test_sphere_diff(self):
        def sphere(x, y):
            z = x ** 2 + y ** 2
            return z
        
        x = nnet.Tensor(np.array(1.0))
        y = nnet.Tensor(np.array(1.0))
        z = sphere(x, y)
        z.backward()
        
        self.assertEqual(x.grad.data, 2.0)
        self.assertEqual(y.grad.data, 2.0)

    def test_2nd_diff(self):
        def f(x):
            y = x**4 - 2*x**2
            return y
        x = nnet.Tensor(np.array(2.0))
        y = f(x)
        y.backward(create_graph=True)
        print(x.grad)
        self.assertEqual(x.grad.data, 24.0)

        gx = x.grad
        gx.backward()
        print(x.grad)
        
        self.assertEqual(x.grad.data, 68.0)

    def test_newton_method(self):
        """Solve arg min f(x)
                  x
           L = f(x + p)^2
             = (f(x) + df(x)/dx*p)^2
           dL/dp = df(x)/dx * (f(x) + df(x)/dx*p) = 0
           df(x)/dx * f(x) = -(df(x)/dx)^2 * p
           p = -f(x) / (df(x)/dx)
        """
        def f(x):
            y = x**4 - 2*x**2
            return y
        
        x = nnet.Tensor(np.array(2.0))
        iters = 10

        for i in range(iters):
            print(i, x)

            # compute y
            y = f(x)
            # compute dy/dx
            x.cleargrad()            
            y.backward(create_graph=True)
            gx = x.grad
            # compute dy/dx^2
            x.cleargrad()
            gx.backward()
            gx2 = x.grad

            x.data -= gx.data / gx2.data

        self.assertEqual(x.data, 1.0)

    def test_double_backprop(self):
        x = nnet.Tensor(np.array(2.0))
        y = x ** 2
        y.backward(create_graph=True)
        gx = x.grad
        x.cleargrad()

        z = gx ** 3 + y
        z.backward()
        print(x.grad)

        self.assertEqual(x.grad.data, 100)

    def test_tensor_matrix(self):
        x0 = np.array([[1, 2, 3], [4, 5, 6]])
        x = nnet.Tensor(x0)
        y = nnet.sin(x)

        print(y)
        np.testing.assert_array_almost_equal(y.data, np.sin(x0))

    def test_reshape(self):
        x = nnet.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        y = nnet.reshape(x, (6,))
        y.backward(retain_grad=True)

        print(x.grad)
        np.testing.assert_array_equal(x.grad.data, [[1, 1, 1], [1, 1, 1]])

        x = nnet.Tensor(np.random.randn(1, 2, 3))
        y = x.reshape((2, 3))        
        self.assertEqual(y.shape, (2, 3))
        print(y)

        y = x.reshape(2, 3)        
        self.assertEqual(y.shape, (2, 3))
        print(y)

    def test_transpose(self):
        x = nnet.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        y = nnet.transpose(x)
        y.backward()

        print(x.grad)
        np.testing.assert_array_equal(x.grad.data, [[1, 1, 1], [1, 1, 1]])

    def test_sum(self):
        x0 = np.array([[1, 2, 3], [4, 5, 6]])
        x = nnet.Tensor(x0)
        y = nnet.sum(x, axis=0)

        np.testing.assert_array_equal(y.data, x0.sum(axis=0))

        y.backward()
        print(y)
        print(x.grad)

        x = nnet.Tensor(np.random.randn(2, 3, 4, 5))
        y = x.sum(keepdims=True)
        print(y)


    def test_matmul(self):
        x = nnet.Tensor(np.random.randn(2, 3))
        W = nnet.Tensor(np.random.randn(3, 4))
        y = nnet.matmul(x, W)
        y.backward()

        print(x.grad.shape)
        print(W.grad.shape)

        np.testing.assert_array_equal(x.grad.shape, (2, 3))
        np.testing.assert_array_equal(W.grad.shape, (3, 4))

    def test_get_item(self):
        x = nnet.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x[1]
        print("get_item()")
        print(y)
        np.testing.assert_array_equal(y.data, [4, 5, 6])
        
        y.backward()
        print(x.grad)
        np.testing.assert_array_equal(x.grad.data, [[0, 0, 0], [1, 1, 1]])

        y = x[:,2]
        print(y)
        np.testing.assert_array_almost_equal(y.data, [3, 6])

    def test_flatten(self):
        x = nnet.Tensor(np.zeros((10, 28, 28, 3)))
        flatten1 = nnet.nn.Flatten(start_dim=1, end_dim=-1)
        y = flatten1(x)
        np.testing.assert_array_equal(y.shape, [10, 28*28*3])

        y.backward()
        np.testing.assert_array_equal(x.grad.data.shape, [10, 28, 28, 3])

        flatten2 = nnet.nn.Flatten(start_dim=1, end_dim=2)
        y = flatten2(x)
        np.testing.assert_array_equal(y.shape, [10, 28*28, 3])

        y.backward()
        np.testing.assert_array_equal(x.grad.data.shape, [10, 28, 28, 3])

if __name__ == "__main__":
    unittest.main()


