import sys
import os.path

from common.log import *
set_log("SCRIPT")

from common.runner import *

import config.configmanager

class Script(object):


    def load_script(name, cmdcenter):
        ''' Load and creates a script from disk '''

        # create script
        script = Script(cmdcenter)
        script.name = name

        # load events
        script.events = configmanager.load_obj("script", name)

        # return object
        return script


    def __init__(self, cmdcenter):
        debug("Creating script")

        self.cmdcenter = cmdcenter
        self.events = []
        self.name = None
        self.current_idx = 0


    def _execute(self):
        ''' Internal execution loop '''

        # main execution loop
        while(self.current_idx < len(self.events) and not self.cmdcenter.env.exit):
            if(self.cmdcenter.time() >= self.events[self.current_idx]["time"]):
                self.cmdcenter.cmd(self.events[self.current_idx]["cmd"])
                self.current_idx += 1


    def start(self):
        ''' Starts the script '''
        debug("Start script")
        async(_execute)


    def add_event(self, event):
        ''' Add an event to the collection of events '''

        # compute insertion index
        idx = [(i == 0 or event["time"] > self.events[i-1]["time"])
               and (i == len(self.events) or event["time"] < self.events[i]["time"])
               for i in xrange(len(self.events) + 1)].index(True)

        # insert event
        self.events.insert(idx, event)

        # increment index if necessary
        if(idx < self.current_idx): self.current_idx += 1



    def save(self, name = None):
        ''' Saves the script '''

        # output events
        configmanager.outp_dict("script", self.events, name)
