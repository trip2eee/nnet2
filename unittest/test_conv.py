import unittest
import numpy as np

import os
import sys
current_dir = os.path.realpath(os.path.curdir)
sys.path.insert(0, current_dir)
import nnet
import nnet.nn.functional as F
import nnet.utils.utils as utils




class ConvTest(unittest.TestCase):
    def test_im2col(self):
        # batch, channel, height, width
        x1 = np.random.rand(1, 3, 7 , 7)

        # (N*OH*OW, C*KH*KW)
        out_size = utils.get_conv_outsize(input_size=7, kernel_size=5, stride=1, pad=0)
        col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
        print(col1.shape)
        np.testing.assert_almost_equal(col1.shape, [1*out_size*out_size, 5*5*3])

        x2 = np.random.rand(10, 3, 7, 7)
        kernel_size = (5, 5)
        stride = (1, 1)
        pad = (0, 0)
        col2 = F.im2col(x2, kernel_size=kernel_size, stride=stride, pad=pad, to_matrix=True)
        print(col2.shape)
        np.testing.assert_almost_equal(col2.shape, [10*out_size*out_size, 5*5*3])

    def test_conv2d(self):
        N, C, H, W = 1, 5, 15, 15
        OC, (KH, KW) = 8, (3, 3)
        x = nnet.Tensor(np.random.randn(N, C, H, W))
        W = np.random.randn(OC, C, KH, KW)
        
        y = F.conv2d(x, W, b=None, stride=1, pad=1)
        y.backward()

        print(y.shape)
        print(x.grad.shape)

        
if __name__ == "__main__":
    unittest.main()
