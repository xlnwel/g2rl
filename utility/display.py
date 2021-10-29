import numpy as np


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return f'\x1b[{";".join(attr)}m{string}\x1b[0m'

def pwc(*args, color='red', bold=False, highlight=False):
    """
    Print with color
    """
    if isinstance(args, (tuple, list)):
        for s in args:
            print(colorize(s, color, bold, highlight))
    else:
        print(colorize(args, color, bold, highlight))

def assert_colorize(cond, err_msg=''):
    assert cond, colorize(err_msg, 'red')

def display_var_info(tf_vars, name='trainable', prefix=''):
    pwc(f'{prefix}{name.title()} variables', color='yellow')
    nparams = 0
    for v in tf_vars:
        name = v.name
        if '/Adam' in name or 'beta1_power' in name or 'beta2_power' in name: continue
        v_params = int(np.prod(v.shape.as_list()))
        nparams += v_params
        if '/b:' in name or '/biases' in name: continue    # Wx+b, bias is not interesting to look at => count params, but not print
        pwc(f'{prefix}   {name}{" "*(100-len(name))} {v_params:d} params {v.shape}', color='yellow')

    pwc(f'{prefix}Total model parameters: {nparams*1e-6:0.4g} million', color='yellow')
	
    return nparams

def display_model_var_info(models):
    learnable_models = {}
    opts = {}
    nparams = 0
    for name, model in models.items():
        if 'target' in name or name in learnable_models or name in opts:
            pass # ignore variables in the target networks
        elif 'opt' in name:
            opts[name] = model
        else:
            learnable_models[name] = model
    
    pwc(f'Learnable models:', color='yellow')
    for name, model in learnable_models.items():
        nparams += display_var_info(
            model.trainable_variables, name=name, prefix='   ')
    pwc(f'Total learnable model parameters: {nparams*1e-6:0.4g} million', color='yellow')
