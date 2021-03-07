''' Optimizer Stuff '''
from torch.optim import SGD, Adam

def get_optim(name, parameters, options):
    if name == "SGD":
        return SGD(parameters, **options)
    elif name == "Adam":
        return Adam(parameters, **options)