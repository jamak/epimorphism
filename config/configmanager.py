import sys
import os.path

import noumena
from config.structs import *

from common.log import *
set_log("CONFIG")


# mappings from config extensions to classes/names
extension_names = {"state": "est", "profile": "prf", "context": "ctx", "environment": "env", "script": "scr"}


def load_dict(type, name, **additional_vars):
    ''' Loads a dictionary into a dictionarized object '''

    vars = load_obj(type, name)
    vars.update(additional_vars)

    # return correct config object
    obj = eval(type.capitalize())(**vars)
    obj.name = name

    return obj


def load_obj(type, name):
    ''' Loads an object. '''
    debug("Loading %s - %s" % (type, name))

    # open file & extract contents
    try:
        file = open("config/" + type + "/" + name + "." + extension_names[type])
        contents = file.read()
        file.close()

    except:
        critical("couldn't read %s - %s" % (type, name))
        return None

    # create & return object
    return eval(contents.replace("\n", ""))


def outp_obj(type, obj, name=None):
    ''' Dumps an object to a file.  Adds newlines after commas for legibility '''
    debug("Saving %s" % type)

    # generate name if necessary
    if(not name):
        i = 0

        while(os.path.exists("config/%s/%s_%d.%s" % (type, type, i, extension_names[type]))):        i += 1
        name = "%s_%d" % (type, i)
        path = "config/%s/%s_%d.%s" % (type, type, i, extension_names[type])

    debug("with name %s" % name)

    # set name if dict
    # if(type(obj) == dict):
    #     obj.name = name

    # open file & dump repr(obj)
    file = open(path, "w")
    file.write(repr(obj).replace(",", ",\n"))
    file.close()

    return name
