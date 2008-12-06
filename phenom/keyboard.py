from OpenGL.GL import *
from OpenGL.GLUT import *

from ctypes import *
from cuda.cuda_defs import *
from copy import *

import Image

from phenom.cmdcenter import *

from common.complex import *


class KeyboardHandler:

    def __init__(self, cmdcenter):
        self.cmdcenter = cmdcenter
        self.state, self.animator, self.engine = self.cmdcenter.state, self.cmdcenter.animator, self.cmdcenter.engine
        self.cmdcenter.renderer.set_keyboard(self.keyboard)


    def keyboard(self, key, x, y):

        if(key == "\033"):
            exit()

        elif(key == "`"):
            if(self.state.manual_iter):
                self.engine.next_frame = True
            self.state.manual_iter = not self.state.manual_iter        

        elif(key == "\040"): # space
            self.engine.next_frame = True            

        elif(key == "\015"): # enter
            StateManager().save_state(self.state, 
                                      Image.frombuffer("RGBA", (self.profile.kernel_dim, self.profile.kernel_dim), self.engine.get_fb(), "raw", "RGBA", 0, 1))

        elif(key == "\\"):
            self.engine.set_fb((float4 * (self.profile.kernel_dim ** 2))())

        elif(key == "1"):
            # self.animator.animate_t("zn[0] * s(z) + zn[1]")
            # self.animator.animate_var("self.state.par[0]", "linear_1d", 200000, {'s':0.0, 'e':2*pi , 'loop':True})
            self.cmdcenter.load_t(0)

        elif(key == "2"):
           self.animator.animate_t("zn[0] * s(z) + zn[1]")

        elif(key in ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";"]):
            i = ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";"].index(key)
            z0 = r_to_p(self.state.zn[i])
            z1 = copy(z0)
            z1[0] += 0.05
            self.animator.animate_var("zn" + str(i), lambda z: self.zn_setter(i, z), "radial_2d", 200, {"s":z0, "e":z1, 'loop':False})

        elif(key in ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/"]):
            i = ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/"].index(key)
            z0 = r_to_p(self.state.zn[i])
            z1 = copy(z0)
            z1[0] -= 0.05
            if(z1[0] < 0.0):
                z1[0] = 0
            self.animator.animate_var("zn" + str(i), lambda z: self.zn_setter(i, z), "radial_2d", 200, {"s":z0, "e":z1, 'loop':False})

        elif(key in ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":"]):
            i = ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":"].index(key)
            z0 = r_to_p(self.state.zn[i])
            z1 = copy(z0)
            z1[1] += 2.0 * pi / 32.0
            self.animator.animate_var("zn" + str(i), lambda z: self.zn_setter(i, z), "radial_2d", 200, {"s":z0, "e":z1, 'loop':False})

        elif(key in ["Z", "X", "C", "V", "B", "N", "M", "<", ">", "?"]):
            i = ["Z", "X", "C", "V", "B", "N", "M", "<", ">", "?"].index(key)
            z0 = r_to_p(self.state.zn[i])
            z1 = copy(z0)
            z1[1] -= 2.0 * pi / 32.0
            self.animator.animate_var("zn" + str(i), lambda z: self.zn_setter(i, z), "radial_2d", 200, {"s":z0, "e":z1, 'loop':False})

        elif(key in ["6", "7", "8", "9", "0", "^", "&", "*", "(", ")"]):
            i = ["6", "7", "8", "9", "0", "^", "&", "*", "(", ")"].index(key)
            x0 = self.state.par[i]
            x1 = self.state.par[i] + 0.05
            self.animator.animate_var("par" + str(i), lambda x: self.par_setter(i, x), "linear_1d", 400, {"s":x0, "e":x1, 'loop':False})

        elif(key in ["y", "u", "i", "o", "p", "Y", "U", "I", "O", "P"]):
            i = ["y", "u", "i", "o", "p", "Y", "U", "I", "O", "P"].index(key)
            x0 = self.state.par[10 + i]
            x1 = self.state.par[10 + i] - 0.05
            self.animator.animate_var("par" + str(10 + i), lambda x: self.par_setter(10 + i, x), "linear_1d", 400, {"s":x0, "e":x1, 'loop':False})


    def zn_setter(self, i, z):
        self.state.zn[i] = z

    def par_setter(self, i, x):
        self.state.par[i] = x


    def do(self):
        messages
