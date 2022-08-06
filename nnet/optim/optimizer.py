class Optimizer:
    """Base optimizer class
    """
    def __init__(self):
        self.target = None
        self.hooks = []     # list of preprocessing functions.

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        # perform preprocessing here.
        for f in self.hooks:
            f(params)
        
        for param in params:
            self.update_one(param)
    
    def update_one(self, param):
        raise NotImplementedError
    
    def add_hook(self, f):
        """This method adds preprocess functions e.g. weight decay, gradient clipping, etc (optional).
        """
        self.hooks.append(f)

        
    