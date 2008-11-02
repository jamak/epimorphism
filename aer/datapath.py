from math import *
from noumena.complex import *

def linear_1d(t, data):
    if(t > 1):
        t = 1
        raise "done"
    return data['s'] * (1 - t) + data['e'] * t


def linear_2d(t, data):
        if(t > 1):
            t = 1
            raise "done"
        return complex(data['s'].real * (1 - t) + data['e'].real * t,
                       data['s'].imag * (1 - t) + data['e'].imag * t)


def radial_2d(t, data):
        if(t > 1):
            t = 1
            raise "done"

        z = [data['s'][0] * (1 - t) + data['e'][0] * t, data['s'][1] * (1 - t) + data['e'][1] * t]

        return p_to_r(z)
                       
