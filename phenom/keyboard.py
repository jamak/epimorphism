from ctypes import *
from cuda.cuda_defs import *

from OpenGL.GL import *
from OpenGL.GLUT import *

from common.complex import *
from common.runner import *

from noumena.config import *

import sys
import time


class KeyboardHandler(object):
    ''' The KeyboardHandler is the GLUT callback that handles keyboard events
        in the Renderer object during normal opperation '''

    def __init__(self, cmdcenter):

        self.cmdcenter = cmdcenter
        self.state, self.context = self.cmdcenter.state, self.cmdcenter.context

        # initialize component list
        self.components = cmdcenter.datamanager.__dict__.keys()
        self.components.sort()


    def keyboard(self, key, x, y):

        # get modifiers
        modifiers = glutGetModifiers()

        # set pars if CTRL
        if((modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL):

            # increment par[i]
            if(ord(key) in [49, 0, 27, 28, 28, 30, 31, 127, 57, 48, 1, 19, 4, 6, 7, 8, 10, 11, 12, 59]): # row 1 & 3
                i = [49, 0, 27, 28, 28, 30, 31, 127, 57, 48, 1, 19, 4, 6, 7, 8, 10, 11, 12, 59].index(ord(key))
                if(modifiers & GLUT_ACTIVE_SHIFT == GLUT_ACTIVE_SHIFT) : i += 20
                x0 = self.state.par[i]
                x1 = self.state.par[i] + 0.05
                self.cmdcenter.linear_1d(self.state.par, i, 400, x0, x1)


            # decrement par[i]
            elif(ord(key) in [17, 23, 5, 18, 20, 25, 21, 9, 15, 16, 26, 24, 3, 22, 2, 14, 13, 44, 46, 31]): # row 2 & 4
                i = [17, 23, 5, 18, 20, 25, 21, 9, 15, 16, 26, 24, 3, 22, 2, 14, 13, 44, 46, 31].index(ord(key))
                if(modifiers & GLUT_ACTIVE_SHIFT == GLUT_ACTIVE_SHIFT) : i += 20
                x0 = self.state.par[i]
                x1 = self.state.par[i] - 0.05
                self.cmdcenter.linear_1d(self.state.par, i, 400, x0, x1)

        else:
            # exit
            if(key == "\033"): # enter
                self.cmdcenter.context.exit = True
                time.sleep(0.1)
                sys.exit()

            # toggle manual iteration
            elif(key == "\011"): # tab
                self.cmdcenter.cmd("manual()")

            # toggle next frame
            elif(key == "\040"): # space
                self.cmdcenter.cmd("next()")

            # save state
            elif(key == "\015"): # enter
                self.cmdcenter.cmd("save()")

            # toggle console
            elif(key == "`"):
                self.cmdcenter.cmd("toggle_console()")

            # reset fb
            elif(key == "\\"):
                self.cmdcenter.cmd("reset_fb()")

            # increment component
            elif(key in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]):
                i = ["1", "2", "3", "4", "5", "6", "7", "8", "9"].index(key)
                run_as_thread(lambda : self.cmdcenter.cmd("inc_data('%s', 1)" % self.components[i]))

            # decrement component
            elif(key in ["q", "w", "e", "r", "t", "y", "u", "i", "o"]):
                i = ["q", "w", "e", "r", "t", "y", "u", "i", "o"].index(key)
                run_as_thread(lambda : self.cmdcenter.cmd("inc_data('%s', -1)" % self.components[i]))

            # increment zn_r
            elif(key in ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";"]):
                i = ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";"].index(key)
                z0 = r_to_p(self.state.zn[i])
                z1 = [z0[0], z0[1]]
                z1[0] += self.context.par_scale * 0.05
                self.cmdcenter.radial_2d(self.state.zn, i, 200, z0, z1)

            # decrement zn_r
            elif(key in ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/"]):
                i = ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/"].index(key)
                z0 = r_to_p(self.state.zn[i])
                z1 = [z0[0], z0[1]]
                z1[0] -= self.context.par_scale * 0.05
                if(z1[0] < 0.0):
                    z1[0] = 0
                self.cmdcenter.radial_2d(self.state.zn, i, 200, z0, z1)

            # increment zn_th
            elif(key in ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":"]):
                i = ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":"].index(key)
                z0 = r_to_p(self.state.zn[i])
                z1 = [z0[0], z0[1]]
                z1[1] += self.context.par_scale * 2.0 * pi / 32.0
                self.cmdcenter.radial_2d(self.state.zn, i, 200, z0, z1)

            # decrement zn_th
            elif(key in ["Z", "X", "C", "V", "B", "N", "M", "<", ">", "?"]):
                i = ["Z", "X", "C", "V", "B", "N", "M", "<", ">", "?"].index(key)
                z0 = r_to_p(self.state.zn[i])
                z1 = [z0[0], z0[1]]
                z1[1] -= self.context.par_scale * 2.0 * pi / 32.0
                self.cmdcenter.radial_2d(self.state.zn, i, 200, z0, z1)

            # magnify par_scale
            elif(key == GLUT_KEY_PAGE_UP):
                self.context.par_scale *= 2.0

            # minify par_scale
            elif(key == GLUT_KEY_PAGE_DOWN):
                self.context.par_scale /= 2.0

            # reset zn
            elif(key == GLUT_KEY_HOME):
                default = ConfigManager().load_dict("default.est")
                for i in xrange(len(default.zn)):
                    self.cmdcenter.radial_2d(self.state.zn, i, 100, r_to_p(self.state.zn[i]), r_to_p(default.zn[i]))

            # reset par
            elif(key == GLUT_KEY_END):
                default = ConfigManager().load_dict("default.est")
                for i in xrange(len(default.par)):
                    self.cmdcenter.linear_1d(self.state.par, i, 100, self.state.par[i], default.par[i])

            # toggle fps
            elif(key == GLUT_KEY_F12):
                self.cmdcenter.cmd("toggle_fps()")

