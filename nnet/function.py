import nnet
import weakref
from nnet.config import Config

class Function:
    def __init__(self):
        pass

    def parameters(self):
        # getattr()?
        raise NotImplementedError

    def __call__(self, *inputs):
        inputs = [nnet.as_tensor(x) for x in inputs]
        
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [nnet.Tensor(y) for y in ys]
                
        if Config.enable_backprop:
            # update generation
            self.generation = max([x.generation for x in inputs])
            
            for output in outputs:
                output.set_creator(self)    # specify the module from which this tensor is computed.
            
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError

