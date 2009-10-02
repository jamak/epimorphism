import noumena

from config.migration import *
from common.complex import *
from phenom.stdmidi import *

import os.path

default_flag = False

class MidiList(list):
    ''' This is an internal class to add midi synchronization to
        changes in parameters. '''

    # maintain copy of origonal setter
    old_set = list.__setitem__

    def __setitem__(self, key, val):

        # set value
        self.old_set(key, val)

        if(hasattr(self, "midi")):

            # lookup bindings
            bindings = self.midi.get_bindings()
            for binding in bindings:

                if(bindings[binding][4] == (self, key)):

                    # compute value
                    f = bindings[binding][2]()
                    f = eval(bindings[binding][3])

                    # send value
                    self.midi.writef(binding, f)


class State(object):
    ''' The State object contains the main configuration parameters for
        the Engine's kernel. '''

    def __init__(self, **vars):

        # update dict with migrated vars
        self.__dict__.update(migrate(vars))

        # set par defaults
        global default_flag
        if(default_flag):
            for i in xrange(len(self.par_names)):
               self.par[i] = float(self.par_defaults[self.par_names[i]])

        # create midi_lists
        self.zn = MidiList(self.zn)
        self.par = MidiList(self.par)


class Profile(object):
    ''' The Profile object contains the configuration settings for the
        Renderer and Engine. '''

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
        the various config settings. '''

    # mappings from config extensions to classes/names
    extension_names = {"state" : "est", "profile" : "prf", "context" : "ctx"}


    def load_dict(self, type, name, **additional_vars):

        # open file & extract contents
        file = open("config/" + type + "/" + name + "." + self.extension_names[type])
<<<<<<< HEAD:config/configmanager.py
=======

>>>>>>> 127da883db27c9a2ddd6d3c85eba550a32b1acac:config/configmanager.py
        contents = file.read()
        file.close()

        # get vars
        vars = eval(contents.replace("\n", ""))
        vars.update(additional_vars)

        # set default_flag
        global default_flag
        default_flag = name == "default"

        # return correct config object
<<<<<<< HEAD:config/configmanager.py
        return eval(type.capitalize())(**vars)
=======
        obj = eval(type.capitalize())(**vars)
        obj.name = name

        return obj
>>>>>>> 127da883db27c9a2ddd6d3c85eba550a32b1acac:config/configmanager.py


    def outp_dict(self, name, obj):

        # open file & dump repr(obj)
        file = open(name, "w")
        file.write(repr(obj.__dict__).replace(",", ",\n"))
        file.close()


    def save_state(self, state, name=None):

        # set correct version
        state.VERSION = noumena.VERSION

        # generate name if necessary
        if(not name):
            i = 0
            while(os.path.exists("config/state/state_%d.est" % i)) : i += 1
            name = "state_" + str(i)

        state.name = name

        # output dict
        self.outp_dict("config/state/%s.est" % name, state)

        # return name
        return name

