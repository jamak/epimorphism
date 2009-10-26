from common.complex import *

get_val    = (lambda x: x)
set_val    = (lambda x, y: y)
get_radius = (lambda z: r_to_p(z)[0])
set_radius = (lambda z, r: p_to_r([r, r_to_p(z)[1]]))
get_th     = (lambda z: r_to_p(z)[1])
set_th     = (lambda z, th: p_to_r([r_to_p(z)[0], th]))
