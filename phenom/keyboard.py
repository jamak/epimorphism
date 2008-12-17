import sys
import time

from OpenGL.GL import *
from OpenGL.GLUT import *

from ctypes import *
from cuda.cuda_defs import *

from common.complex import *

from common.runner import *

class KeyboardHandler(object):

    def __init__(self, cmdcenter):
        self.cmdcenter = cmdcenter
        self.state, self.animator, self.engine, self.renderer, self.context = self.cmdcenter.state, self.cmdcenter.animator, self.cmdcenter.engine, self.cmdcenter.renderer, self.cmdcenter.context


    def keyboard(self, key, x, y):
        modifiers = glutGetModifiers()

        if(modifiers & GLUT_ACTIVE_CTRL == GLUT_ACTIVE_CTRL):
            if(ord(key) in [49, 0, 27, 28, 28, 30, 31, 127, 57, 48, 1, 19, 4, 6, 7, 8, 10, 11, 12, 59]):
                i = [49, 0, 27, 28, 28, 30, 31, 127, 57, 48, 1, 19, 4, 6, 7, 8, 10, 11, 12, 59].index(ord(key))
                if(modifiers & GLUT_ACTIVE_SHIFT == GLUT_ACTIVE_SHIFT):
                    i += 20
                x0 = self.state.par[i]
                x1 = self.state.par[i] + 0.05
                self.animator.animate_var("par" + str(i), self.cmdcenter.par_set_i(i), "linear_1d", 400, {"s":x0, "e":x1, 'loop':False})
            elif(ord(key) in [17, 23, 5, 18, 20, 25, 21, 9, 15, 16, 26, 24, 3, 22, 2, 14, 13, 44, 46, 31]):
                i = [17, 23, 5, 18, 20, 25, 21, 9, 15, 16, 26, 24, 3, 22, 2, 14, 13, 44, 46, 31].index(ord(key))
                if(modifiers & GLUT_ACTIVE_SHIFT == GLUT_ACTIVE_SHIFT):
                    i += 20
                x0 = self.state.par[i]
                x1 = self.state.par[i] - 0.05
                self.animator.animate_var("par" + str(i), self.cmdcenter.par_set_i(i), "linear_1d", 400, {"s":x0, "e":x1, 'loop':False})
            return

        if(key == "\033"):
            self.cmdcenter.context.midi = False
            time.sleep(0.1)
            sys.exit()

        elif(key == "`"):
            self.renderer.toggle_console()

        elif(key == "\011"): # tab
            if(self.context.manual_iter):
                self.context.next_frame = True
            self.context.manual_iter = not self.context.manual_iter

        elif(key == "\040"): # space
            self.context.next_frame = True

        elif(key == "\015"): # enter
            self.cmdcenter.save()

        elif(key == "\\"):
            self.engine.reset_fb()

        elif(key in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]):
            i = ["1", "2", "3", "4", "5", "6", "7", "8", "9"].index(key)
            keys = self.cmdcenter.datamanager.__dict__.keys()
            keys.sort()
            run_as_thread(lambda : self.cmdcenter.inc_data(keys[i], 1))

        elif(key in ["q", "w", "e", "r", "t", "y", "u", "i", "o"]):
            i = ["q", "w", "e", "r", "t", "y", "u", "i", "o"].index(key)
            keys = self.cmdcenter.datamanager.__dict__.keys()
            keys.sort()
            run_as_thread(lambda : self.cmdcenter.inc_data(keys[i], -1))

        elif(key in ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";"]):
            i = ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";"].index(key)
            z0 = r_to_p(self.state.zn[i])
            z1 = [z0[0], z0[1]]
            z1[0] += self.context.par_scale * 0.05
            self.animator.animate_var("zn" + str(i),self.cmdcenter.zn_set_i(i), "radial_2d", 200, {"s":z0, "e":z1, 'loop':False})

        elif(key in ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/"]):
            i = ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/"].index(key)
            z0 = r_to_p(self.state.zn[i])
            z1 = [z0[0], z0[1]]
            z1[0] -= self.context.par_scale * 0.05
            if(z1[0] < 0.0):
                z1[0] = 0
            self.animator.animate_var("zn" + str(i),self.cmdcenter.zn_set_i(i), "radial_2d", 200, {"s":z0, "e":z1, 'loop':False})

        elif(key in ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":"]):
            i = ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":"].index(key)
            z0 = r_to_p(self.state.zn[i])
            z1 = [z0[0], z0[1]]
            z1[1] += self.context.par_scale * 2.0 * pi / 32.0
            self.animator.animate_var("zn" + str(i),self.cmdcenter.zn_set_i(i), "radial_2d", 200, {"s":z0, "e":z1, 'loop':False})

        elif(key in ["Z", "X", "C", "V", "B", "N", "M", "<", ">", "?"]):
            i = ["Z", "X", "C", "V", "B", "N", "M", "<", ">", "?"].index(key)
            z0 = r_to_p(self.state.zn[i])
            z1 = [z0[0], z0[1]]
            z1[1] -= self.context.par_scale * 2.0 * pi / 32.0
            self.animator.animate_var("zn" + str(i),self.cmdcenter.zn_set_i(i), "radial_2d", 200, {"s":z0, "e":z1, 'loop':False})

        elif(key == GLUT_KEY_PAGE_UP):
            self.context.par_scale *= 2.0

        elif(key == GLUT_KEY_PAGE_DOWN):
            self.context.par_scale /= 2.0

        elif(key == GLUT_KEY_F12):
            self.renderer.toggle_fps()

