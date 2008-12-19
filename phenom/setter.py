from common.complex import *


class Setter(object):
    ''' Returns various closures for setting zn & par. '''

    # return closures for getting state.zn[i]
    def zn_get_i(self, i):
        return lambda : self.state.zn[i]

    def zn_get_r_i(self, i):
        return lambda : r_to_p(self.state.zn[i])[0]

    def zn_get_th_i(self, i):
        return lambda : r_to_p(self.state.zn[i])[1]


    # closures for s etting state.zn[i]
    def zn_set_i(self, i):
        def zn_set(i, z):
            self.state.zn[i] = z
        return lambda z: zn_set(i, z)

    def zn_set_r_i(self, i):
        def zn_set_r(i, r):
            p = r_to_p(self.state.zn[i])
            p[0] = r
            self.state.zn[i] = p_to_r(p)
        return lambda r: zn_set_r(i, r)

    def zn_set_th_i(self, i):
        def zn_set_th(i, th):
            p = r_to_p(self.state.zn[i])
            p[1] = th
            self.state.zn[i] = p_to_r(p)
        return lambda th: zn_set_th(i, th)


    # closure for getting state.par[i]
    def par_get_i(self, i):
        return lambda : self.state.par[i]

    # closure for setting state.par[i]
    def par_set_i(self, i):
        def par_set(i, x):
            self.state.par[i] = x
        return lambda x: par_set(i, x)

