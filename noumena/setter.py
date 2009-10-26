from common.complex import *

def get_val(x):       return x
def set_val(x, y):    return y
def get_radius(z):    return r_to_p(z)[0]
def set_radius(z, r): return p_to_r([r, r_to_p(z)[1]])
def get_th(z):        return r_to_p(z)[1]
def set_th(z, th):    return p_to_r([r_to_p(z)[0], th])
