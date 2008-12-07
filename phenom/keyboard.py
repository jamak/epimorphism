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
        self.state, self.animator, self.engine, self.renderer = self.cmdcenter.state, self.cmdcenter.animator, self.cmdcenter.engine, self.cmdcenter.renderer


    def keyboard(self, key, x, y):

        if(key == "\033"):
            exit()

        elif(key == "`"):
            self.renderer.toggle_console()

        elif(key == "\011"): # tab
            if(self.state.manual_iter):
                self.engine.next_frame = True
            self.state.manual_iter = not self.state.manual_iter        

        elif(key == "\040"): # space
            self.engine.next_frame = True            

        elif(key == "\015"): # enter
            image = Image.frombuffer("RGBA", (self.engine.profile.kernel_dim, self.engine.profile.kernel_dim), self.engine.get_fb(), "raw", "RGBA", 0, 1)

            StateManager().save_state(self.state, image)

        elif(key == "\\"):
            self.engine.reset_fb()

        elif(key == "!"):
            self.cmdcenter.inc_t(1)

        elif(key == "@"):
            self.cmdcenter.inc_t_seed(1)

        elif(key == "Q"):
            self.cmdcenter.inc_t(-1)

        elif(key == "W"):
            self.cmdcenter.inc_t_seed(-1)

        elif(key == "#"):
            self.cmdcenter.cmd("self.a=7")

        elif(key == "E"):
            self.cmdcenter.cmd("print self.a")

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

        elif(key in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "^", "&", "*", "(", ")"]):
            i = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "^", "&", "*", "(", ")"].index(key)
            x0 = self.state.par[i]
            x1 = self.state.par[i] + 0.05
            self.animator.animate_var("par" + str(i), lambda x: self.par_setter(i, x), "linear_1d", 400, {"s":x0, "e":x1, 'loop':False})

        elif(key in ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "Y", "U", "I", "O", "P"]):
            i = ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "Y", "U", "I", "O", "P"].index(key)
            x0 = self.state.par[i]
            x1 = self.state.par[i] - 0.05
            self.animator.animate_var("par" + str(i), lambda x: self.par_setter(i, x), "linear_1d", 400, {"s":x0, "e":x1, 'loop':False})


    def zn_setter(self, i, z):
        self.state.zn[i] = z

    def par_setter(self, i, x):
        self.state.par[i] = x


    def do(self):
        messages
