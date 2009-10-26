import sys
import os.path

from common.log import *
set_log("SCRIPT")

from common.runner import *

import config.configmanager


class Script(object):
    ''' Contains a timestamped sequence of commands which are executed in the Cmd environment '''

    def load_script(name, cmdcenter):
        ''' Load and creates a script from disk '''

        # create script
        script = Script(cmdcenter)
        script.name = name

        # load events

        # return object
        return script


    def __init__(self, cmdcenter, name = None):
        debug("Creating script")

        self.cmdcenter, self.cmdcenter = cmdcenter, name

        self.events = (self.name and configmanager.load_obj("script", name)) or []
        self.current_idx = 0


    def _execute(self):
        ''' Internal execution loop '''

        # main execution loop
        while(self.current_idx < len(self.events) and not self.cmdcenter.env.exit):
            if(self.cmdcenter.time() >= self.events[self.current_idx]["time"]):
                debug("Cmd from script %s" % (self.name or ""))
                self.cmdcenter.cmd(self.events[self.current_idx]["cmd"])
                self.current_idx += 1


    def start(self):
        ''' Starts the script '''
        debug("Start script")

        async(_execute)


    def add_event(self, time, cmd):
        ''' Add an event to the collection of events '''
        debug("Adding event at %f", %f)

        # compute insertion index
        idx = [(i == 0 or time > self.events[i-1]["time"])
               and (i == len(self.events) or time < self.events[i]["time"])
               for i in xrange(len(self.events) + 1)].index(True)

        # insert event
        self.events.insert(idx, {"time":time, "cmd":cmd})

        # increment index if necessary
        if(idx < self.current_idx): self.current_idx += 1


    def save(self, name = None):
        ''' Saves the script '''

        # output events
        self.name = configmanager.outp_obj("script", self.events, self.name)
