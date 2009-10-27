import threading
import pypm
import re

import time

from common.complex import *
from noumena.setter import *

from noumena.mididevices import *

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

        # load bindings
        self.bindings = eval(self.context.midi_controller)

        # set default bindings
        self.binding_idx = 0

        # send defaults
        self.send_bindings()

        # init thread
        threading.Thread.__init__(self)


    def output_binding(self, binding_id):
        ''' Evaluates and outputs value of a binding '''

        binding = self.bindings[self.binding_idx][binding_id]
        f = ((eval("get_" + binding[2])(self.cmdcenter.get_val(binding[0], binding[1]))) - binding[3][1]) / binding[3][0]

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
            if(self.binding_idx == i) : self.writef(0, 65 + i, 1.0)


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

                # compute & output value
                f = binding[3][0] * f + binding[3][1]

                old = self.cmdcenter.get_val(binding[0], binding[1])
                val = eval("set_" + binding[2])(old, f)

                #self.cmdcenter.cmd('radial_2d(zn, %d, 0.08, %s, %s)' % (binding[1], str(r_to_p(old)), str(r_to_p(val))))
                self.cmdcenter.set_val(val, binding[0], binding[1])

            # change bindings - HACK: buttons switch bindings
            elif(channel >= 65 and channel <= 72):

                self.binding_idx = channel - 65
                if val == 0 : self.binding_idx = 0
                self.send_bindings()

