import noumena

from noumena.migration import *

import os.path



class State(object):
    ''' The State object is the main configuration parameters for
        the Engine's kernel.  '''

    def __init__(self, **vars):

        # update dict with migrated vars
        self.__dict__.update(migrate(vars))

        # set par defaults
        for i in xrange(len(self.par_names)):
            self.par[i] = float(self.par_defaults[self.par_names[i]])


class Profile(object):
    ''' The Profile object is the configuration settings for the
        Renderer and Engine '''

    def __init__(self, **vars):

        # init
        self.__dict__.update(vars)


class Context(object):
    ''' The Context object encapsulates the variables defining the
        current execution context of the application. '''

    def __init__(self, **vars):

        # init
        self.__dict__.update(vars)


class ConfigManager(object):
    ''' The ConfigManager class is responsible for managing(save/load)
        the various config settings '''


    # mappings from config extensions to classes/names
    extension_names = {"est" : "state", "prf" : "profile", "ctx" : "context"}


    def load_dict(self, name, **additional_vars):

        # get extension
        extension = name.split(".")[1]

        # open file & extract contents
        file = open("config/" + self.extension_names[extension] + "/" + name)
        contents = file.read()
        file.close()

        # get vars
        vars = eval(contents.replace("\n", ""))
        vars.update(additional_vars)

        # return correct config object
        return eval(self.extension_names[extension].capitalize())(**vars)


    def outp_dict(self, name, obj):

        # open file & dump repr(obj)
        file = open(name, "w")
        file.write(repr(obj.__dict__).replace(",", ",\n"))
        file.close()


    def save_state(self, state, name=""):

        # set correct version
        state.VERSION = noumena.VERSION

        # generate name if necessary
        if(name == ""):
            i = 0
            while(os.path.exists("config/state/state_%d.est" % i)) : i += 1
            name = "state_" + str(i)

        # output dict
        self.outp_dict("config/state/%s.est" % name, state)

        # return name
        return name

