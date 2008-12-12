from ctypes import *
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

        file = open("state/" + state_name + ".est", "r")
        return State(**eval(file.read().replace("\n", "")))


    def load_profile(self, profile_name, **vars):
        log("st: load profile - " + profile_name)
        log("st: with vars - " + str(vars))

        file = open("noumena/profile/" + profile_name + ".prf", "r")
        return Profile(**eval(file.read().replace("\n", "")))


    def default_context(self, **vars):
        vars.update({'par_scale' : 1, 'midi' : False, 'server' : False, 'render_video' : True, 'max_video_frames' : 20, 'video_frame_time' : 100 })
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
        file.write(repr(state.__dict__).replace(",", ",\n"))
        file.close()


    def save_profile(self, profile, name):
        log("st: save profile")
        log("st:   as " + name)

        file = open("noumena/profile/" + name + ".prf", "w")
        file.write(repr(profile.__dict__).replace(",", ",\n"))
        file.close()


