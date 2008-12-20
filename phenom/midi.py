import threading
import pypm
import re

import time

from common.complex import *
from phenom.stdmidi import *
from phenom.setter import *

class MidiHandler(threading.Thread, Setter):
    ''' The MidiHandler object is a threaded object to handle midi input
        events and to send midi output information '''

    def __init__(self, cmdcenter):

        self.cmdcenter, self.state = cmdcenter, cmdcenter.state

        # find devices
        for loop in range(pypm.CountDevices()):\

            interf,name,inp,outp,opened = pypm.GetDeviceInfo(loop)

            if(re.compile("BCF2000").search(name) and inp == 1):
                self.input_device = loop
            if(re.compile("BCF2000").search(name) and outp == 1):
                self.output_device = loop

        # open devices
        try:
            self.midi_in = pypm.Input(self.input_device)
            self.midi_out = pypm.Output(self.output_device, 10)
        except:
            print "MIDI device not found"
            self.cmdcenter.context.midi = False

        # create sefault zn bindings
        self.bindings0 = {81: [self.cmdcenter.zn_set_r_i(0),  "m1(f)", self.cmdcenter.zn_get_r_i(0),  "m1_inv(f)", (self.state.zn, 0)],
                          82: [self.cmdcenter.zn_set_r_i(1),  "m0(f)", self.cmdcenter.zn_get_r_i(1),  "m0_inv(f)", (self.state.zn, 1)],
                          83: [self.cmdcenter.zn_set_r_i(2),  "m0(f)", self.cmdcenter.zn_get_r_i(2),  "m0_inv(f)", (self.state.zn, 2)],
                          84: [self.cmdcenter.zn_set_r_i(3),  "m0(f)", self.cmdcenter.zn_get_r_i(3),  "m0_inv(f)", (self.state.zn, 3)],
                          85: [self.cmdcenter.zn_set_r_i(4),  "m0(f)", self.cmdcenter.zn_get_r_i(4),  "m0_inv(f)", (self.state.zn, 4)],
                          86: [self.cmdcenter.zn_set_r_i(5),  "m0(f)", self.cmdcenter.zn_get_r_i(5),  "m0_inv(f)", (self.state.zn, 5)],
                          87: [self.cmdcenter.zn_set_r_i(6),  "m1(f)", self.cmdcenter.zn_get_r_i(6),  "m1_inv(f)", (self.state.zn, 6)],
                          88: [self.cmdcenter.zn_set_r_i(7),  "m0(f)", self.cmdcenter.zn_get_r_i(7),  "m0_inv(f)", (self.state.zn, 7)]}
        self.bindings0.update(dict([(1 + i, [self.cmdcenter.zn_set_th_i(i), "m4(f)", self.cmdcenter.zn_get_th_i(i), "m4_inv(f)", (self.state.zn, i)]) for i in xrange(8)]))

        self.bindings1 = dict([(1 + i, [self.cmdcenter.par_set_i(i), "m0(f)", self.cmdcenter.par_get_i(i), "m0_inv(f)", (self.state.par, i)]) for i in xrange(8, 10)])
        self.bindings1.update(dict([(1 + i, [self.cmdcenter.zn_set_th_i(i), "m4(f)", self.cmdcenter.zn_get_th_i(i), "m4_inv(f)", (self.state.zn, i)]) for i in xrange(8, 10)]))

        # create par bindings
        self.bindings2 = dict([(81 + i, [self.cmdcenter.par_set_i(i), "m0(f)", self.cmdcenter.par_get_i(i), "m0_inv(f)", (self.state.par, i)]) for i in xrange(8)])

        self.bindings3 = dict([(81 + i, [self.cmdcenter.par_set_i(i + 8), "m0(f)", self.cmdcenter.par_get_i(i + 8), "m0_inv(f)", (self.state.par, i)]) for i in xrange(8)])

        self.bindings4 = dict([(81 + i, [self.cmdcenter.par_set_i(i + 16), "m0(f)", self.cmdcenter.par_get_i(i + 16), "m0_inv(f)", (self.state.par, i)]) for i in xrange(8)])

        # set default bindings
        self.bindings = 0

        # send defaults
        self.send_bindings()

        # init thread
        threading.Thread.__init__(self)


    def send_bindings(self):

        # send all bindings
        bindings = self.get_bindings()

        for key in bindings:
            binding = bindings[key]
            f = binding[2]()
            f = eval(binding[3])
            self.writef(key, f)

        # send binding buttons
        for i in xrange(0, 8):
            self.writef(65 + i, 0.0)
            if(self.bindings == i) : self.writef(65 + i, 1.0)


    def writef(self, channel, f):

        # get val
        val = int(f * 128.0)
        if(val == 128): val = 127

        # send
        self.midi_out.Write([[[176, channel, val, 0], pypm.Time()]])


    def get_bindings(self):

        # return bindings
        return getattr(self, "bindings%d" % self.bindings)


    def run(self):

        # run loop
        while(True and self.cmdcenter.context.midi):

            # sleep / exit
            while(not self.midi_in.Poll() and not self.cmdcenter.context.exit) : time.sleep(0.01)
            if(self.cmdcenter.context.exit) : exit()

            # read
            data = self.midi_in.Read(1)

            # set vars
            channel = data[0][0][1]
            val = data[0][0][2]

            # print channel, " ", val

            # get f
            f = val / 128.0
            if(val == 127.0) : f = 1.0

            # check bindings
            bindings = self.get_bindings()
            if(bindings.has_key(channel)):

                # process binding
                binding = bindings[channel]
                binding[0](eval(binding[1]))

            # change bindings
            elif(channel >= 65 and channel <= 72):

                self.bindings = channel - 65
                if val == 0 : self.bindings = 0
                self.send_bindings()

