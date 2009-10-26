from config.migration import *


class MidiList(list):
    ''' This is an internal class to add midi synchronization to
        changes in parameters. '''

    # maintain copy of origonal setter
    old_set = list.__setitem__

    def __setitem__(self, key, val):
        # set value
        self.old_set(key, val)

        if(hasattr(self, "midi")):
            self.midi.mirror(self, key)


class State(object):
    ''' Configuration parameters for generating Frames. '''

    def __init__(self, **vars):
        # update dict with migrated vars
        self.__dict__.update(migrate(vars))

        # create midi_lists
        self.zn  = MidiList(self.zn)
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
