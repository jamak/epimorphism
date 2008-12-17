from common.complex import *

class Setter(object):
    def zn_get_i(self, i):
        return lambda : self.state.zn[i]

    def zn_get_r_i(self, i):
        return lambda : r_to_p(self.state.zn[i])[0]

    def zn_get_th_i(self, i):
        return lambda : r_to_p(self.state.zn[i])[1]


    def zn_set(self, i, z):
        self.state.zn[i] = z

    def zn_set_r(self, i, r):
        p = r_to_p(self.state.zn[i])
        p[0] = r
        self.state.zn[i] = p_to_r(p)

    def zn_set_th(self, i, th):
        p = r_to_p(self.state.zn[i])
        p[1] = th
        self.state.zn[i] = p_to_r(p)


    def zn_set_i(self, i):
        return lambda z: self.zn_set(i, z)

    def zn_set_r_i(self, i):
        return lambda r: self.zn_set_r(i, r)

    def zn_set_th_i(self, i):
        return lambda th: self.zn_set_th(i, th)



    def par_get_i(self, i):
        return lambda : self.state.par[i]


    def par_set(self, i, x):
        self.state.par[i] = x

    def par_set_i(self, i):
        return lambda x: self.par_set(i, x)

