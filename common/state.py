#
#  state.py - definition and management of State/Profile/Context
#
#    The epimorphism project uses the three structures state/profile/context#    for configuration:
#      state   - configuration variable for the systme being rendered
#      profile - internal renderer/engine configuration
#      context - execution context & misc parameters
#

import noumena

import os.path

from common.migration import *



class State(object):

    def __init__(self, **vars):
        self.__dict__.update(migrate(vars))



class Profile(object):

    def __init__(self, **vars):
        self.__dict__.update(vars)



class Context(object):

    def __init__(self, **vars):
        self.__dict__.update(vars)



class StateManager(object):

    extension_names = {"est" : "state", "prf" : "profile", "ctx" : "context"}

    def __init__(self):
        self.shiz = self.load_dict

    def load_dict(self, name, **additional_vars):
        extension = name.split(".")[1]
        file = open("common/" + self.extension_names[extension] + "/" + name)
        vars = eval(file.read().replace("\n", ""))
        vars.update(additional_vars)
        file.close()
        return eval(self.extension_names[extension].capitalize())(**vars)


    def outp_dict(self, name, obj):
        file = open(name, "w")
        file.write(repr(obj.__dict__).replace(",", ",\n"))
        file.close()


    def save_state(self, state, image, name=""):
        state.VERSION = noumena.VERSION

        if(name == ""):
            i = 0
            while(os.path.exists("common/state/state_" + str(i) + ".est")):
                i += 1
            name = "state_" + str(i)

        if(image):
            image.save("common/state/image_" + str(i) + ".png")

        self.outp_dict("common/state/" + name + ".est", state)

