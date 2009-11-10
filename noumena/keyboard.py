from ctypes import *
from cuda.cuda_defs import *

from OpenGL.GL import *
from OpenGL.GLUT import *

from common.complex import *
from common.runner import *
from config import configmanager

import sys
import time

from common.log import *
set_log("KEYBOARD")

class KeyboardHandler(object):
    ''' The KeyboardHandler is the GLUT callback that handles keyboard events
        in the Renderer object during normal opperation '''

    def __init__(self, cmdcenter, context):

        self.cmdcenter = cmdcenter
        self.state, self.context = self.cmdcenter.state, context

        #initialize component list
        self.components = cmdcenter.componentmanager.component_list()


    def keyboard(self, key, x, y):

        # get modifiers
        modifiers = glutGetModifiers()

        async(lambda : eval("self." + self.context.keyboard)(key, modifiers))


    def common(self, key, modifiers):
        # exit
        if(key == '|'): # escape
            self.cmdcenter.env.exit = True

        # toggle console
        elif(key == "`"):
            self.cmdcenter.cmd("toggle_console()")

        # reset fb
        elif(key == "\\"):
            self.cmdcenter.cmd("reset_fb()")

        # reset zn
        elif(key == GLUT_KEY_HOME):
            default = configmanager.load_dict("state", "default")
            for i in xrange(len(default.zn)):
                self.cmdcenter.cmd('radial_2d(zn, %d, 0.4, %s, %s)' % (i, str(r_to_p(self.state.zn[i])), str(r_to_p(default.zn[i]))))

        # reset par
        elif(key == GLUT_KEY_END):
            default = configmanager.load_dict("state", "default")
            for i in xrange(len(default.par)):
                self.cmdcenter.cmd('linear_1d(par, %d, 0.4, %f, %f)' % (i, self.state.par[i], default.par[i]))

        # tap tempo
        elif(key == GLUT_KEY_F1):
            self.cmdcenter.cmd("tap_tempo()")

        # toggle echo
        elif(key == GLUT_KEY_F11):
            self.cmdcenter.cmd("toggle_echo()")

        # toggle fps
        elif(key == GLUT_KEY_F12):
            self.cmdcenter.cmd("toggle_fps()")

        # save state
        elif(key == "\015"): # enter
            self.cmdcenter.cmd("save()")

        # toggle manual iteration
        elif(key == "\011"): # tab
            self.cmdcenter.cmd("manual()")

        # toggle next frame
        elif(key == "\040"): # space
            self.cmdcenter.cmd("next()")


    def full(self, key, modifiers):

        self.common(key, modifiers)

        # set pars if CTRL
        if((modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL):

            # increment par[i]
            if(ord(key) in [49, 0, 27, 28, 28, 30, 31, 127, 57, 48, 1, 19, 4, 6, 7, 8, 10, 11, 12, 59]): # row 1 & 3
                i = [49, 0, 27, 28, 28, 30, 31, 127, 57, 48, 1, 19, 4, 6, 7, 8, 10, 11, 12, 59].index(ord(key))
                if(modifiers & GLUT_ACTIVE_SHIFT == GLUT_ACTIVE_SHIFT) : i += 20
                x0 = self.state.par[i]
                x1 = self.state.par[i] + 0.05
                self.cmdcenter.cmd('linear_1d(par, %d, kbd_switch_spd, %f, %f)' % (i, x0, x1))

            # decrement par[i]
            elif(ord(key) in [17, 23, 5, 18, 20, 25, 21, 9, 15, 16, 26, 24, 3, 22, 2, 14, 13, 44, 46, 31]): # row 2 & 4
                i = [17, 23, 5, 18, 20, 25, 21, 9, 15, 16, 26, 24, 3, 22, 2, 14, 13, 44, 46, 31].index(ord(key))
                if(modifiers & GLUT_ACTIVE_SHIFT == GLUT_ACTIVE_SHIFT) : i += 20
                x0 = self.state.par[i]
                x1 = self.state.par[i] - 0.05
                self.cmdcenter.cmd('linear_1d(par, %d, kbd_switch_spd, %f, %f)' % (i, x0, x1))

        else:

            # increment component
            if(key in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]):
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
                self.cmdcenter.cmd('radial_2d(zn, %d, kbd_switch_spd, %s, %s)' % (i, str(z0), str(z1)))

            # decrement zn_r
            elif(key in ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/"]):
                i = ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/"].index(key)
                z0 = r_to_p(self.state.zn[i])
                z1 = [z0[0], z0[1]]
                z1[0] -= self.context.par_scale * 0.05
                if(z1[0] < 0.0):
                    z1[0] = 0
                self.cmdcenter.cmd('radial_2d(zn, %d, kbd_switch_spd, %s, %s)' % (i, str(z0), str(z1)))

            # increment zn_th
            elif(key in ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":"]):
                i = ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":"].index(key)
                z0 = r_to_p(self.state.zn[i])
                z1 = [z0[0], z0[1]]
                z1[1] += self.context.par_scale * 2.0 * pi / 32.0
                self.cmdcenter.cmd('radial_2d(zn, %d, kbd_switch_spd, %s, %s)' % (i, str(z0), str(z1)))

            # decrement zn_th
            elif(key in ["Z", "X", "C", "V", "B", "N", "M", "<", ">", "?"]):
                i = ["Z", "X", "C", "V", "B", "N", "M", "<", ">", "?"].index(key)
                z0 = r_to_p(self.state.zn[i])
                z1 = [z0[0], z0[1]]
                z1[1] -= self.context.par_scale * 2.0 * pi / 32.0
                self.cmdcenter.cmd('radial_2d(zn, %d, kbd_switch_spd, %s, %s)' % (i, str(z0), str(z1)))

            # magnify par_scale
            elif(key == GLUT_KEY_PAGE_UP):
                self.context.par_scale *= 2.0

            # minify par_scale
            elif(key == GLUT_KEY_PAGE_DOWN):
                self.context.par_scale /= 2.0



    def live(self, key, modifiers):

        self.common(key, modifiers)

        multiplier = 1

        if((modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT):
            multiplier = 2

        if((modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL):
            multiplier = 0

        # switch_midi
        if(key == GLUT_KEY_F10):
            if(self.context.midi_controller[1] == "BCF_LIVE"):
                info("Switch to BCF_FULL bindings")
                self.context.midi_controller[1] = "BCF_FULL"
                self.cmdcenter.interface.midi.load_bindings()
            elif(self.context.midi_controller[1] == "BCF_FULL"):
                info("Switch to BCF_LIVE bindings")
                self.context.midi_controller[1] = "BCF_LIVE"
                self.cmdcenter.interface.midi.load_bindings()

        elif(key == "1"):
            self.cmdcenter.eventmanager.switch_component("T", multiplier)
        elif(key == "2"):
            self.cmdcenter.eventmanager.switch_component("T_SEED", multiplier)
        elif(key == "3"):
            self.cmdcenter.eventmanager.switch_component("SEED_W", multiplier)
        elif(key == "4"):
            self.cmdcenter.eventmanager.switch_component("SEED_WT", multiplier)
        elif(key == "5"):
            self.cmdcenter.eventmanager.switch_component("SEED_A", multiplier)
        elif(key == "6"):
            self.cmdcenter.eventmanager.switch_component("REDUCE", multiplier)

        elif(key == "q"):
            self.cmdcenter.eventmanager.rotate90(0, multiplier)
        elif(key == "w"):
            self.cmdcenter.eventmanager.rotate90(2, multiplier)
        elif(key == "e"):
            self.cmdcenter.eventmanager.rotate90(8, multiplier)
        elif(key == "r"):
            self.cmdcenter.eventmanager.rotate90(10, multiplier)


