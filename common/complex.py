from math import *

def r_to_p(z):
    return [abs(z), atan2(z.imag, z.real)]

def p_to_r(z):
    return complex(z[0] * cos(z[1]), z[0] * sin(z[1]))
