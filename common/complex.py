from math import *


# convert rect to polar
def r_to_p(z):
    arg = atan2(z.imag, z.real)
    if(arg < 0) : arg += 2.0 * 3.14159
    return [abs(z), arg]


# conver polar to rect
def p_to_r(z):
    return complex(abs(z[0]) * cos(z[1]), abs(z[0]) * sin(z[1]))
