from ast import Param
import nnet.tensor
import weakref

class Parameter(nnet.tensor.Tensor):
    pass

class Module:
    def __init__(self):
        self._params = set()
    
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
            

