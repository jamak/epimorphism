import sys
import os.path

import noumena

from config.migration import *

from config.structs import *

from common.log import *
set_log("CONFIG")

class ConfigManager(object):
    ''' The ConfigManager class is responsible for managing(save/load)
        the various config settings. '''


    # mappings from config extensions to classes/names
    extension_names = {"state": "est", "profile": "prf", "context": "ctx", "environment": "env", "script": "scr"}


    def load_obj(type, name):
        ''' Loads an object. '''
        debug("Loading %s - %s" % (type, name))

        # open file & extract contents
        try:
            file = open("config/" + type + "/" + name + "." + ConfigManager.extension_names[type])
            contents = file.read()
            file.close()

        except:
            exception("couldn't read %s - %s" % (type, name))
            sys.exit()

        # create & return object
        return eval(contents.replace("\n", ""))


    def load_dict(type, name, **additional_vars):
        ''' Loads a dictionary into a dictionarized object '''

        vars = ConfigManager.load_obj(type, name)
        vars.update(additional_vars)

        # return correct config object
        obj = eval(type.capitalize())(**vars)
        obj.name = name

        return obj


    def outp_obj(type, obj, name=None):
        ''' Dumps an object to a file.  Adds newlines after commas for legibility '''
        debug("Saving %s" % type)

        # generate name if necessary
        if(not name):
            i = 0
            while(os.path.exists("config/%s/%s_%d.%s" % (type, type, i, ConfigManager.extension_names[type]))) : i += 1
            name = "%s_%d" % (type, i)

        debug("with name %s" % name)

        # set name if dict
        if(type(obj) == dict):
            obj.name = name

        # open file & dump repr(obj)
        file = open(name, "w")
        file.write(repr(obj).replace(",", ",\n"))
        file.close()

        return name
