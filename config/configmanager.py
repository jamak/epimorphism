import sys
import os.path

import noumena

from config.migration import *

from common.log import *
set_log("CONFIG")


class MidiList(list):
    ''' This is an internal class to add midi synchronization to
        changes in parameters. '''

    # maintain copy of origonal setter
    old_set = list.__setitem__

    def __setitem__(self, key, val):
        # set value
        self.old_set(key, val)

        #if(hasattr(self, "midi")):
        #    self.midi.mirror(self, key)


class State(object):
    ''' Configuration parameters for generating Frames. '''

    def __init__(self, **vars):
        # update dict with migrated vars
        self.__dict__.update(migrate(vars))

        # create midi_lists
        self.zn = MidiList(self.zn)
        self.par = MidiList(self.par)


class Profile(object):
    ''' Configuration settings for the Engine. '''

    def __init__(self, **vars):
        # init
        self.__dict__.update(vars)


class Context(object):
    ''' Configuration settings for the Interface. '''

    def __init__(self, **vars):
        # init
        self.__dict__.update(vars)


class Environment(object):
    ''' Configuration settings for the application. '''

    def __init__(self, **vars):
        # init
        self.__dict__.update(vars)


class ConfigManager(object):
    ''' The ConfigManager class is responsible for managing(save/load)
        the various config settings. '''


    # mappings from config extensions to classes/names
    extension_names = {"state": "est", "profile": "prf", "context": "ctx", "environment": "env"}


    def load_dict(self, type, name, **additional_vars):
        ''' loads a dictionary into a dictionarized object '''

        debug("loading %s - %s" % (type, name))

        # open file & extract contents
        try:
            file = open("config/" + type + "/" + name + "." + self.extension_names[type])
            contents = file.read()
            file.close()

        except:
            exception("couldn't read %s - %s" % (type, name))
            sys.exit()

        # get vars
        vars = eval(contents.replace("\n", ""))
        vars.update(additional_vars)

        # return correct config object
        obj = eval(type.capitalize())(**vars)
        obj.name = name

        return obj


    def outp_dict(self, name, obj):
        ''' dumps a dict to a file '''

        debug("serializing dictionary %s" % name)

        # open file & dump repr(obj)
        file = open(name, "w")
        file.write(repr(obj.__dict__).replace(",", ",\n"))
        file.close()


    def save_state(self, state, name=None):
        ''' saves a state to disk '''

        debug("saving state")

        # set correct version
        state.VERSION = noumena.VERSION

        # generate name if necessary
        if(not name):
            i = 0
            while(os.path.exists("config/state/state_%d.est" % i)) : i += 1
            name = "state_" + str(i)

        state.name = name

        debug("with name %s" % name)

        # output dict
        self.outp_dict("config/state/%s.est" % name, state)

        # return name
        return name

