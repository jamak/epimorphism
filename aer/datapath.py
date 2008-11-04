from math import *
from noumena.complex import *

def linear_1d(t, data):
    if(t > 1):
        t = data['loop'] and fmod(t, 1.0) or 1
    return (data['s'] * (1 - t) + data['e'] * t, t != 1 or data['loop'])


def linear_2d(t, data):
    if(t > 1):
        t = data['loop'] and fmod(t, 1.0) or 1
    return (complex(data['s'].real * (1 - t) + data['e'].real * t,
                    data['s'].imag * (1 - t) + data['e'].imag * t), t != 1 or data['loop'])


def radial_2d(t, data):
    if(t > 1):
        t = data['loop'] and fmod(t, 1.0) or 1
    z = [data['s'][0] * (1 - t) + data['e'][0] * t, data['s'][1] * (1 - t) + data['e'][1] * t]
    return (p_to_r(z), t != 1 or data['loop'])
                       

def wave_1d(t, data):
    return (data['a'] * sin(2.0 * pi * t + data['th']) + data['b'], True)


