from common.complex import *


# clamp  a <= z_r <= b
def clamp_r(zn, i, a, b):
    p = r_to_p(zn[i])
    p[0] = min(max(p[0], a), b)
    zn[i] = p_to_r(p)


# min  a <= z_r
def min_r(zn, i, a):

    p = r_to_p(zn[i])
    p[0] = max(p[0], a)
    zn[i] = p_to_r(p)


# list of all default functions
default_funcs = {"clamp_r" : clamp_r, "min_r" : min_r}
