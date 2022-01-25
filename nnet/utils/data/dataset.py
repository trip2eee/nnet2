import numpy as np
from nnet.utils.data.transforms import *
from nnet import Tensor
import matplotlib.pyplot as plt
import gzip
from nnet.utils.utils import get_file, cache_dir

class Dataset:
    def __init__(self):
        self.x = None
        self.y = None
    
    def __getitem__(self, index):
        raise NotImplementedError
        
    def __len__(self):
        raise NotImplementedError
    
class Spiral(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super(Spiral, self).__init__()

        x, y = self.get_spiral(train=train)
        self.x = x
        self.y = y

        self.x = Tensor(self.x)
        self.y = Tensor(self.y)
        
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x
    
    def __getitem__(self, index):
        x = self.transform(self.x[index])
        y = self.target_transform(self.y[index])
        return x, y

    def __len__(self):
        return len(self.x)

    def get_spiral(self, train=True):
        seed = 1984 if train else 2020
        np.random.seed(seed=seed)

        num_data, num_class, input_dim = 100, 3, 2
        data_size = num_class * num_data
        x = np.zeros((data_size, input_dim), dtype=np.float32)
        y = np.zeros(data_size, dtype=np.int)

        for j in range(num_class):
            for i in range(num_data):
                rate = i / num_data
                radius = 1.0 * rate
                theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
                ix = num_data * j + i
                x[ix] = np.array([radius * np.sin(theta),
                                radius * np.cos(theta)]).flatten()
                y[ix] = j
        # Shuffle
        indices = np.random.permutation(num_data * num_class)
        x = x[indices]
        y = y[indices]
        return x, y


# =============================================================================
# MNIST-like dataset: MNIST / CIFAR /
# =============================================================================
class MNIST(Dataset):

    def __init__(self, train=True, transform=None):
        super(MNIST, self).__init__()
        self.train = train
        self.transform = transform

        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.x = self._load_data(data_path)
        self.x = self.x.astype(np.float32) / 255.0
        self.y = self._load_label(label_path)

        self.x = Tensor(self.x)
        self.y = Tensor(self.y)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def __getitem__(self, index):
        x = self.x[index]
        if self.transform is not None:
            x = self.transfrom(x)

        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.x[
                    np.random.randint(0, len(self.x) - 1)].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}