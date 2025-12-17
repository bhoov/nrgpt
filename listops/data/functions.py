
def median(*inputs):
    """return the median of the inputs"""
    sorted_inputs = sorted(inputs)
    n = len(sorted_inputs)
    # we need the output to be integers, so: 
    return sorted_inputs[n // 2]
    # if n % 2 == 1:
    #     return sorted_inputs[n // 2]
    # else:
    #     return (sorted_inputs[n // 2 - 1] + sorted_inputs[n // 2]) / 2
    

def prod(*inputs):
    p = 1
    for i in inputs:
        p *= i
    return p

def add(*inputs):
    """sum the inputs and return the result modulo mod"""
    return sum(inputs)

def nadd(*inputs):
    """alternatingly + and - the inputs"""
    out,s =0,1
    for i in range(len(inputs)):
        out += inputs[i] * s
        s *= -1
    return out 

def parmax(*l):
    m = max(l)
    return m if (m % 2) == 0 else m-1

def get_funcs_mod(funs, mod):
    """return list of functions with name func_{mod} for each function in funs"""
    # return [lambda x, f=f: f(x, mod) for f in funs]
    # we want the functions to have aname func_{mod}, so we can't use lambda
    # we need to define a function for each function in funs
    out_f = []
    for f in funs:
        def func(*inputs, f=f):
            return f(*inputs) % mod
        func.__name__ = f"{f.__name__}_{mod}"
        out_f.append(func)
    return out_f
    
