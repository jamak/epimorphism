from ctypes import *
from cuda.cuda_defs import *

from OpenGL.GL import *
from OpenGL.GLUT import *

from common.complex import *
from common.runner import *

from config.configmanager import *

import sys
import time


class KeyboardHandler(object):
    ''' The KeyboardHandler is the GLUT callback that handles keyboard events
        in the Renderer object during normal opperation '''

    def __init__(self, cmdcenter):

        self.cmdcenter = cmdcenter
        self.state, self.context = self.cmdcenter.state, self.cmdcenter.context

        #initialize component list
        self.components = cmdcenter.componentmanager.component_list()


    def keyboard(self, key, x, y):

        # get modifiers
        modifiers = glutGetModifiers()


        async(lambda : eval("self." + self.context.keyboard_func)(key, modifiers))


    def standard(self, key, modifiers):
        # set pars if CTRL
        if((modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL):

            # increment par[i]
            if(ord(key) in [49, 0, 27, 28, 28, 30, 31, 127, 57, 48, 1, 19, 4, 6, 7, 8, 10, 11, 12, 59]): # row 1 & 3
                i = [49, 0, 27, 28, 28, 30, 31, 127, 57, 48, 1, 19, 4, 6, 7, 8, 10, 11, 12, 59].index(ord(key))
                if(modifiers & GLUT_ACTIVE_SHIFT == GLUT_ACTIVE_SHIFT) : i += 20
                x0 = self.state.par[i]
                x1 = self.state.par[i] + 0.05
                self.cmdcenter.linear_1d(self.state.par, i, self.context.kbd_switch_spd, x0, x1)

            # decrement par[i]
            elif(ord(key) in [17, 23, 5, 18, 20, 25, 21, 9, 15, 16, 26, 24, 3, 22, 2, 14, 13, 44, 46, 31]): # row 2 & 4
                i = [17, 23, 5, 18, 20, 25, 21, 9, 15, 16, 26, 24, 3, 22, 2, 14, 13, 44, 46, 31].index(ord(key))
                if(modifiers & GLUT_ACTIVE_SHIFT == GLUT_ACTIVE_SHIFT) : i += 20
                x0 = self.state.par[i]
                x1 = self.state.par[i] - 0.05
                self.cmdcenter.linear_1d(self.state.par, i, self.context.kbd_switch_spd, x0, x1)

        else:
            # exit
            if(key == "\033"): # escape
                self.cmdcenter.context.exit = True

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


            # for testing
            elif(key == "}"):
                self.cmdcenter.load(0)

            # increment component
            elif(key in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]):
                i = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"].index(key)

                self.cmdcenter.cmd("inc_data('%s', 1)" % self.components[i])

            # decrement component
            elif(key in ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"]):
                i = ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"].index(key)
                self.cmdcenter.cmd("inc_data('%s', -1)" % self.components[i])

            # increment zn_r
            elif(key in ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";"]):
                i = ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";"].index(key)
                z0 = r_to_p(self.state.zn[i])
                z1 = [z0[0], z0[1]]
                z1[0] += self.context.par_scale * 0.05
                self.cmdcenter.radial_2d(self.state.zn, i, self.context.kbd_switch_spd, z0, z1)

             # decrement zn_r
            elif(key in ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/"]):
                i = ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/"].index(key)
                z0 = r_to_p(self.state.zn[i])
                z1 = [z0[0], z0[1]]
                z1[0] -= self.context.par_scale * 0.05
                if(z1[0] < 0.0):
                    z1[0] = 0
                self.cmdcenter.radial_2d(self.state.zn, i, self.context.kbd_switch_spd, z0, z1)

            # increment zn_th
            elif(key in ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":"]):
                i = ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":"].index(key)
                z0 = r_to_p(self.state.zn[i])
                z1 = [z0[0], z0[1]]
                z1[1] += self.context.par_scale * 2.0 * pi / 32.0
                self.cmdcenter.radial_2d(self.state.zn, i, self.context.kbd_switch_spd, z0, z1)

            # decrement zn_th
            elif(key in ["Z", "X", "C", "V", "B", "N", "M", "<", ">", "?"]):
                i = ["Z", "X", "C", "V", "B", "N", "M", "<", ">", "?"].index(key)
                z0 = r_to_p(self.state.zn[i])
                z1 = [z0[0], z0[1]]
                z1[1] -= self.context.par_scale * 2.0 * pi / 32.0
                self.cmdcenter.radial_2d(self.state.zn, i, self.context.kbd_switch_spd, z0, z1)

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
                    self.cmdcenter.radial_2d(self.state.zn, i, 0.4, r_to_p(self.state.zn[i]), r_to_p(default.zn[i]))

            # reset par
            elif(key == GLUT_KEY_END):
                default = ConfigManager().load_dict("default.est")
                for i in xrange(len(default.par)):
                    self.cmdcenter.linear_1d(self.state.par, i, 0.4, self.state.par[i], default.par[i])

            # toggle fps
            elif(key == GLUT_KEY_F12):
                self.cmdcenter.cmd("toggle_fps()")

            # toggle echo
            elif(key == GLUT_KEY_F11):
                self.cmdcenter.cmd("toggle_echo()")




    def bm2009(self, key, modifiers):

        if((modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL):
            self.cmdcenter.moduleCmd('bm2009', 'set_var', {'var':'switch_exponent', 'val':2})

        if((modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT):
            self.cmdcenter.moduleCmd('bm2009', 'set_var', {'var':'switch_exponent', 'val':-1})

        if(key == '|'): # escape
            self.cmdcenter.context.exit = True

        # toggle console
        elif(key == "`"):
            self.cmdcenter.cmd("toggle_console()")

        # reset fb
        elif(key == "\\"):
            self.cmdcenter.cmd("reset_fb()")

        # reset zn
        elif(key == GLUT_KEY_HOME):
            default = ConfigManager().load_dict("default.est")
            for i in xrange(len(default.zn)):
                self.cmdcenter.radial_2d(self.state.zn, i, 0.4, r_to_p(self.state.zn[i]), r_to_p(default.zn[i]))

        # reset par
        elif(key == GLUT_KEY_END):
            default = ConfigManager().load_dict("default.est")
            for i in xrange(len(default.par)):
                self.cmdcenter.linear_1d(self.state.par, i, 0.4, self.state.par[i], default.par[i])

        # switch events
        elif(key == GLUT_KEY_F1):
            self.cmdcenter.moduleCmd('bm2009', 'tap_tempo', {})

        # switch events
        elif(key == GLUT_KEY_F10):
            self.cmdcenter.moduleCmd('bm2009', 'switch_events', {})

        # toggle echo
        elif(key == GLUT_KEY_F11):
            self.cmdcenter.cmd("toggle_echo()")

        # toggle fps
        elif(key == GLUT_KEY_F12):
            self.cmdcenter.cmd("toggle_fps()")

        elif(key == "1"):
            self.cmdcenter.moduleCmd('bm2009', 'switch_t', {})

        elif(key == "2"):
            self.cmdcenter.moduleCmd('bm2009', 'switch_t_seed', {})

        elif(key == "3"):
            self.cmdcenter.moduleCmd('bm2009', 'switch_seed_w', {})

        elif(key == "4"):
            self.cmdcenter.moduleCmd('bm2009', 'switch_reduce', {})

        elif(key == "5"):
            self.cmdcenter.moduleCmd('bm2009', 'switch_seed_a', {})

        elif(key == "6"):
            self.cmdcenter.moduleCmd('bm2009', 'switch_seed_wt', {})



