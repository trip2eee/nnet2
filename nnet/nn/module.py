from ast import Param
import nnet.tensor
import weakref
import numpy as np
import os

class Parameter(nnet.tensor.Tensor):
    pass

class Module:
    def __init__(self):
        self._params = set()
        self.train_mode = True  # training mode.
    
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Module)):
            self._params.add(name)
        elif isinstance(value, (list)):
            for i, item in enumerate(value):
                if isinstance(item, (Parameter, Module)):
                    setattr(self, name + "_" + str(i), item)

        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inptus):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Module):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()
    
    def to_cpu(self):
        for param in self.params():
            param.to_cpu()
    
    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def to(self, device):
        if device == 'gpu':
            self.to_gpu()
        else:
            self.to_cpu()

    def _flatten_params(self, params_dict, parent_key=''):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Module):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj
    
    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise
    
    def load_weights(self, path):
        
        npz = np.load(path, allow_pickle=True)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]
    
    def train(self, mode=True):
        """This method sets module in training mode.
           mode (bool) - whether to set training mode (True) or evaluation mode (False). Default: True.
        """
        self.train_mode = mode
        # change train mode recursively.
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Module):
                obj.train(mode)
    
    def eval(self):
        """This method sets the module in evaluation mode.
        """
        self.train(False)
