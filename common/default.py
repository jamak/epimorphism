# defaults

from common.complex import *

def clamp_r(var, a, b):
    p = r_to_p(var)
    p[0] = min(max(p[0], a), b)
    return p_to_r(p)

def min_r(var, a):
    p = r_to_p(var)
    p[0] = max(p[0], a)
    return p_to_r(p)

default_funcs = {"clamp_r":clamp_r, "min_r":min_r}
