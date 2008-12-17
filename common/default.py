from common.complex import *


# clamp  a <= z_r <= b
def clamp_r(z, a, b):
    p = r_to_p(z)
    p[0] = min(max(p[0], a), b)
    return p_to_r(p)


# min  a <= z_r
def min_r(z, a):
    p = r_to_p(z)
    p[0] = max(p[0], a)
    return p_to_r(p)


# list of all default functions
default_funcs = {"clamp_r" : clamp_r, "min_r" : min_r}
