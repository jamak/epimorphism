from ctypes import *
import pickle
import os.path

from common.logger import *


class State:
    def __init__(self, **vars):
        self.__dict__.update(vars)


class Profile:
    def __init__(self, **vars):
        self.__dict__.update(vars)

class Context:
    def __init__(self, **vars):
        self.__dict__.update(vars)


class StateManager:
    def load_state(self, state_name, **vars):
        log("st: load state - " + state_name)
        log("st: with vars - " + str(vars))

        state = State(manual_iter=False, FRACT=5, T="zn[0] * z + zn[1]", T_SEED="zn[6] * z + zn[7]", SEED="seed_wca", COLORIFY="rotate_hsls",  REDUCE="torus_reduce",
                      par=[0.0 for i in range(40)], zn=[complex(0,0) for i in range(10)], short_damping = 10, vp_scale=1.0, vp_center_x=0.0, vp_center_y=0.0,
                      par_names={"_SEED_W" : 0, "_SEED_W_BASE" : 1, "_SEED_W_THRESH" : 2, "_COLOR_DHUE" : 3, "_CULL_DEPTH" : 4, "_COLOR_A" : 5, "_COLOR_S" : 6, "_COLOR_V" : 7,
                                 "_COLOR_TH_EFF" : 8},
                      SEED_W="lines_box", SEED_C="simple_color", SEED_A="circular_alpha", **vars)
        state.zn[0] = complex(1.0, 0)
        state.zn[6] = complex(1.0, 0)

        state.par[0] = 0.1
        state.par[1] = -1.0
        state.par[5] = 1.0
        state.par[6] = 1.0
        state.par[7] = 1.0


        self.save_state(state, None, "default")
        return state

        file = open("state/" + state_name + ".est", "r")
        state = pickle.load(file)
        state.__dict__.update(**vars)
        return state


    def load_profile(self, profile_name, **vars):
        log("st: load profile - " + profile_name)
        log("st: with vars - " + str(vars))

        file = open("noumena/profile/" + profile_name + ".prf", "r")
        profile = pickle.load(file)
        profile.__dict__.update(**vars)
        return profile

        #profile = Profile(name=profile_name, viewport_width=1680, viewport_height=1050, full_screen=True, viewport_refresh=60, vp_scale=1.0, vp_center_x=0.0,
        #                  vp_center_y=0.0, kernel_dim=2048, debug_freq=125.0)
        #self.save_profile(profile, "lcd1")
        #return profile


    def default_context(self, **vars):
        vars.update({'par_scale' : 1, 'midi' : False, 'server' : False, 'render_movie' : False})
        return Context(**vars)

    def save_state(self, state, image, name=''):
        log("st: save state")

        if(name == ''):
            i = 0
            while(os.path.exists("state/state_" + str(i) + ".est")):
                i += 1
            name = "state_" + str(i)

        log("st:   as " + name)

        if(image):
            image.save("state/image_" + str(i) + ".png")

        file = open("state/" + name + ".est", "w")
        pickle.dump(state, file)


    def save_profile(self, profile, name):
        log("st: save profile")
        log("st:   as " + name)

        file = open("noumena/profile/" + name + ".prf", "w")
        pickle.dump(profile, file)

