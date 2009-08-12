import threading
import pypm
import re

import time

from common.complex import *
from phenom.stdmidi import *
from phenom.setter import *

class MidiHandler(threading.Thread, Setter):
    ''' The MidiHandler object is a threaded object that handles midi input
        events and that sends midi output information '''

    def __init__(self, cmdcenter):

        self.cmdcenter, self.state = cmdcenter, cmdcenter.state

        # find devices
        for loop in range(pypm.CountDevices()):\

            interf,name,inp,outp,opened = pypm.GetDeviceInfo(loop)
            print name

            #if(re.compile("BCF2000").search(name) and inp == 1):
            #    self.input_device = loop

            #if(re.compile("BCF2000").search(name) and outp == 1):
            #    self.output_device = loop

            if(re.compile("UC-33").search(name) and inp == 1):
                self.input_device = loop

            if(re.compile("UC-33").search(name) and outp == 1):
                self.output_device = loop

        # open devices
        try:
            self.midi_in = pypm.Input(self.input_device)
            self.midi_out = pypm.Output(self.output_device, 10)
            print "Found MIDI device"
        except:
            self.midi_in = None
            self.midi_out = None
            print "MIDI device not found"
            self.cmdcenter.context.midi = False

        # create sefault zn bindings
        self.bindings0 = {81: [self.zn_set_r_i(0),  "m1(f)", self.zn_get_r_i(0),  "m1_inv(f)", (self.state.zn, 0)],
                          82: [self.zn_set_r_i(1),  "m0(f)", self.zn_get_r_i(1),  "m0_inv(f)", (self.state.zn, 1)],
                          83: [self.zn_set_r_i(2),  "m1(f)", self.zn_get_r_i(2),  "m1_inv(f)", (self.state.zn, 2)],
                          84: [self.zn_set_r_i(3),  "m0(f)", self.zn_get_r_i(3),  "m0_inv(f)", (self.state.zn, 3)],
                          85: [self.zn_set_r_i(4),  "m0(f)", self.zn_get_r_i(4),  "m0_inv(f)", (self.state.zn, 4)],
                          86: [self.zn_set_r_i(5),  "m0(f)", self.zn_get_r_i(5),  "m0_inv(f)", (self.state.zn, 5)],
                          87: [self.zn_set_r_i(6),  "m0(f)", self.zn_get_r_i(6),  "m0_inv(f)", (self.state.zn, 6)],
                          88: [self.zn_set_r_i(7),  "m0(f)", self.zn_get_r_i(7),  "m0_inv(f)", (self.state.zn, 7)]}
        self.bindings0.update(dict([(1 + i, [self.zn_set_th_i(i), "m4(f)", self.zn_get_th_i(i), "m4_inv(f)", (self.state.zn, i)]) for i in xrange(8)]))

        self.bindings1 = {81: [self.zn_set_r_i(8),   "m1(f)", self.zn_get_r_i(8),   "m1_inv(f)", (self.state.zn, 8)],
                          82: [self.zn_set_r_i(9),   "m0(f)", self.zn_get_r_i(9),   "m0_inv(f)", (self.state.zn, 9)],
                          83: [self.zn_set_r_i(10),  "m1(f)", self.zn_get_r_i(10),  "m1_inv(f)", (self.state.zn, 10)],
                          84: [self.zn_set_r_i(11),  "m0(f)", self.zn_get_r_i(11),  "m0_inv(f)", (self.state.zn, 11)],
                          85: [self.zn_set_r_i(12),  "m0(f)", self.zn_get_r_i(12),  "m0_inv(f)", (self.state.zn, 12)],
                          86: [self.zn_set_r_i(13),  "m0(f)", self.zn_get_r_i(13),  "m0_inv(f)", (self.state.zn, 13)],
                          87: [self.zn_set_r_i(14),  "m0(f)", self.zn_get_r_i(14),  "m0_inv(f)", (self.state.zn, 14)],
                          88: [self.zn_set_r_i(15),  "m0(f)", self.zn_get_r_i(15),  "m0_inv(f)", (self.state.zn, 15)]}
        self.bindings1.update(dict([(1 + i, [self.zn_set_th_i(8 + i), "m4(f)", self.zn_get_th_i(8 + i), "m4_inv(f)", (self.state.zn, 8 + i)]) for i in xrange(8)]))

        # create par bindings
        self.bindings2 = dict([(81 + i, [self.par_set_i(i),      "m0(f)", self.par_get_i(i),      "m0_inv(f)", (self.state.par, i)])      for i in xrange(8)])

        self.bindings3 = dict([(81 + i, [self.par_set_i(i + 8),  "m0(f)", self.par_get_i(i + 8),  "m0_inv(f)", (self.state.par, i)])      for i in xrange(8)])

        self.bindings4 = dict([(81 + i, [self.par_set_i(i + 16), "m0(f)", self.par_get_i(i + 16), "m0_inv(f)", (self.state.par, i + 16)]) for i in xrange(8)])

        self.bindings5 = dict([(81 + i, [self.par_set_i(i + 24), "m0(f)", self.par_get_i(i + 24), "m0_inv(f)", (self.state.par, i + 24)]) for i in xrange(8)])

        self.bindings6 = dict([(81 + i, [self.par_set_i(i + 32), "m0(f)", self.par_get_i(i + 32), "m0_inv(f)", (self.state.par, i + 32)]) for i in xrange(8)])

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
        if(self.midi_out):
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

            # get f
            f = val / 128.0
            if(val == 127.0) : f = 1.0

            print "MIDI", channel, " ", val, f

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

