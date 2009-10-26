import threading
import pypm
import re

import time

from common.complex import *
from noumena.stdmidi import *
from phenom.setter import *

import noumena.midibindings

from common.log import *
set_log("MIDI")


class MidiHandler(threading.Thread):
    ''' The MidiHandler object is a threaded object that handles midi input
        events and that sends midi output information '''


    def __init__(self, cmdcenter, context):
        self.cmdcenter, self.context = cmdcenter, context

        # find devices - MAYBE A BIT FLAKEY
        for i in range(pypm.CountDevices()):

            interf,name,inp,outp,opened = pypm.GetDeviceInfo(i)
            if(not re.compile("Midi Through Port|TiMidity").search(name)):
                debug("ID: %d INTERFACE: %s NAME: %s %s OPENED? %s" % (i, interf, name, (inp == 1 and "INPUT" or "OUTPUT"), str(opened)))

            if(re.compile(self.context.midi_controller).search(name) and inp == 1):
                self.input_device = i
                break

            if(re.compile(self.context.midi_controller).search(name) and outp == 1):
                self.output_device = i

        # open devices
        try:
            self.midi_in = pypm.Input(self.input_device)
            self.midi_out = pypm.Output(self.output_device, 10)
            info("Found MIDI device")
        except:
            self.midi_in = None
            self.midi_out = None
            info("MIDI device not found")
            self.context.midi = False


        # create device bindings
        self.BCF2000 = []
        self.BCF2000 = [{(0, 81): ["state.zn",  0,  "radius", (1.0, 1.0)],
                         (0, 82): ["state.zn",  1,  "radius", (1.0, 0.0)],
                         (0, 83): ["state.zn",  2,  "radius", (1.0, 1.0)],
                         (0, 84): ["state.zn",  3,  "radius", (1.0, 0.0)],
                         (0, 85): ["state.zn",  8,  "radius", (1.0, 1.0)],
                         (0, 86): ["state.zn",  9,  "radius", (1.0, 0.0)],
                         (0, 87): ["state.zn",  10, "radius", (1.0, 1.0)],
                         (0, 88): ["state.zn",  11, "radius", (1.0, 0.0)],
                         (0, 1) : ["state.par", 0,  "val",    (0.4, 0.2)],  # seed_w
                         (0, 2) : ["state.par", 1,  "val",    (1.0, 0.0)],  # color_phi
                         (0, 3) : ["state.par", 2,  "val",    (1.0, 0.0)],  # colod_psi
                         (0, 4) : ["state.par", 3,  "val",    (0.6, 0.4)],  # color_a
                         (0, 5) : ["state.par", 4,  "val",    (0.8, 0.2)],  # color_s
                         (0, 6) : ["state.zn",  8,  "th",     (3.14, 0.0)], # zn[th][8]
                         (0, 7) : ["state.par", 7,  "val",    (0.6, 0.0)],  # seed_w_thresh
                         (0, 8) : ["state.par", 18, "val",    (1.0, 0.0)],  # colod_dhue
                         }]

        # load bindings
        self.bindings = getattr(self, self.context.midi_controller)

        # set default bindings
        self.binding_idx = 0

        # send defaults
        self.send_bindings()

        # init thread
        threading.Thread.__init__(self)


    def output_binding(self, binding_id):
        ''' Evaluates and outputs value of a binding '''

        binding = self.bindings[self.binding_idx][binding_id]

        f = eval("get_" + binding[2])((self.cmdcenter.get_val(binding[0], binding[1]) - binding[3][1]) / binding[3][0])

        # print binding_id, binding, s, f
        self.writef(binding_id[0], binding_id[1], f)


    def mirror(self, obj, key):
        ''' Echos a change in obj to a midi controller'''

        # lookup bindings
        bindings = self.bindings[self.binding_idx]

        # send correct binding - a bit weird
        for binding_id, binding in bindings.items():
            if(eval("self.cmdcenter.%s" % binding[0]) == obj and binding[1] == key):
                self.output_binding(binding_id)


    def send_bindings(self):
        ''' Send all bindings to midi controller '''

        # lookup bindings
        bindings = self.bindings[self.binding_idx]

        # send all bindings
        for binding_id in bindings:
            self.output_binding(binding_id)

        # send binding buttons - HACK: for switching bindings
        for i in xrange(0, 8):
            self.writef(0, 65 + i, 0.0)
            if(self.bindings == i) : self.writef(0, 65 + i, 1.0)


    def writef(self, bank, channel, f):
        ''' Write a value out '''

        # get val
        val = int(f * 128.0)
        if(val == 128): val = 127

        # send
        if(self.midi_out):
            self.midi_out.Write([[[176 + bank, channel, val, 0], pypm.Time()]])


    def run(self):
        ''' Main execution loop '''

        # run loop
        while(True and self.context.midi):

            # sleep / exit
            while(not self.midi_in.Poll() and not self.cmdcenter.env.exit) : time.sleep(0.01)
            if(self.cmdcenter.env.exit) : exit()

            # read
            data = self.midi_in.Read(1)

            # set vars
            bank = data[0][0][0] % 16
            channel = data[0][0][1]
            val = data[0][0][2]

            # get f
            f = val / 128.0
            if(val == 127.0) : f = 1.0

            # print "MIDI", bank, channel, val, f

            # check bindings
            bindings = self.bindings[self.binding_idx]
            if(bindings.has_key((bank, channel))):
                binding = bindings[(bank, channel)]

                # compute & outpu value
                f = binding[3][0] * f + binding[3][1]
                val = eval("set_" + binding[2])(self.cmdcenter.get_val(binding[0], binding[1]), f)

                self.cmdcenter.set_val(val, binding[0], binding[1])

            # change bindings - HACK: buttons switch bindings
            elif(channel >= 65 and channel <= 72):

                self.bindings = channel - 65
                if val == 0 : self.bindings = 0
                self.send_bindings()

