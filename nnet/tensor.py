import numpy as np
import nnet
import nnet.config

try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)


class Tensor:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if isinstance(data, array_types):
                self.data = data
            elif np.isscalar(data):
                self.data = np.array(data)
            else:
                raise TypeError("{} is not supported".format(type(data)))
        else:
            self.data = None    # for None parameters e.g. bias of layers with no bias used.

        self.grad = None
        self.creator = None
        self.generation = 0
        self.name = name
    
    def set_creator(self, module):
        """
        This method specifies the instance of the module from which this tensor is computed.
        """
        self.creator = module
        self.generation = module.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        """ backward propagation.
            retain_grad (bool) if this flag is true, interim results of gradient computation is preserved. Otherwise, the interim resuls are deleted to save memory.
            create_graph (bool) if this flag is true, inputs and outputs are saved in forward propagation of Module class.
                                The forward propagation of Module class is also called in computation of gradient in backward propagation of Tensor class.
        """
        if self.grad is None:
            xp = nnet.cuda.get_array_module(self.data)
            self.grad = Tensor(xp.ones_like(self.data))

        """
        Backward propagation
        
        forward:  x -> module -> y
        backward: gx (dL/dx) <- module <- gy (dL/dy)
        dL/dx = dL/dy*dy/dx
        """
        modules = []
        seen_set = set()

        def add_module(mod):
            if mod is not None and mod not in seen_set:
                modules.append(mod)
                seen_set.add(mod)
                modules.sort(key=lambda x: x.generation)
            
        add_module(self.creator)

        while modules:
            mod = modules.pop()
            x = mod.inputs
            y = mod.outputs

            gys = [output().grad for output in mod.outputs]

            with nnet.config.using_config('enable_backprop', create_graph):
                gxs = mod.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                
                for x, gx in zip(mod.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_module(x.creator)

                if not retain_grad:
                    for y in mod.outputs:
                        y().grad = None

    def cleargrad(self):
        self.grad = None
            

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
        
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'tensor(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 7)
        return 'tensor(' + p + ')'

    def __mul__(self, other):
        return nnet.mul(self, other)

    def __rmul__(self, other):
        return nnet.mul(self, other)
    
    def __add__(self, other):
        return nnet.add(self, other)

    def __radd__(self, other):
        return nnet.add(self, other)

    def __neg__(self):
        return nnet.neg(self)

    def __sub__(self, other):
        return nnet.sub(self, other)
    
    def __rsub__(self, other):
        return nnet.rsub(self, other)

    def __truediv__(self, other):
        return nnet.div(self, other)
    
    def __rtruediv__(self, other):
        return nnet.rdiv(self, other)

    def __pow__(self, c):
        return nnet.pow(self, c)

    def __getitem__(self, slices):
        return nnet.nn.functional.get_item(self, slices)

    # def __eq__(self, other):
    #     if isinstance(other, np.ndarray):
    #         comp = self.data == other
    #     else:
    #         comp = self.data == other.data

    #     return Tensor(comp)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return nnet.nn.functional.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        
        return nnet.nn.functional.transpose(self, axes)

    def max(self, axis=None, keepdims=False):
        return nnet.nn.functional.max(self, axis, keepdims)

    def min(self, axis=None, keepdims=False):
        return nnet.nn.functional.min(self, axis, keepdims)

    @property
    def T(self):
        return nnet.nn.functional.transpose(self)
        
    def sum(self, axis=None, keepdims=False):
        return nnet.sum(self, axis, keepdims)

    def to_cpu(self):
        if self.data is not None:
            self.data = nnet.cuda.as_numpy(self.data)
    
    def to_gpu(self):
        if self.data is not None:
            self.data = nnet.cuda.as_cupy(self.data)
    
    def to(self, device):
        if device == 'gpu':
            self.to_gpu()
        else:
            self.to_cpu()
    
    def numpy(self):
        """This method returns the numpy ndarray of the self tensor.
        """
        return self.data

