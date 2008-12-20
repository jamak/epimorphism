import threading
import pypm
import re

import time

from common.complex import *
from phenom.stdmidi import *


class MidiHandler(threading.Thread):
    ''' The MidiHandler object is a threaded object to handle midi input
        events and to send midi output information '''

    def __init__(self, cmdcenter):

        self.cmdcenter, self.state = cmdcenter, cmdcenter.state

        # find devices
        for loop in range(pypm.CountDevices()):
            interf,name,inp,outp,opened = pypm.GetDeviceInfo(loop)
            if(re.compile("BCF2000").search(name) and inp == 1):
                self.input_device = loop
            if(re.compile("BCF2000").search(name) and outp == 1):
                self.output_device = loop

        # open devicesx
        try:
            self.midi_in = pypm.Input(self.input_device)
            self.midi_out = pypm.Output(self.output_device, 10)
        except:
            print "MIDI device not found"
            self.cmdcenter.context.midi = False


        # create sefault zn bindings
        self.bindings1 = {81: [self.cmdcenter.zn_set_r_i(0),  "m1(f)", self.cmdcenter.zn_get_r_i(0),  "m1_inv(f)", (self.state.zn, 0)],
                          82: [self.cmdcenter.zn_set_r_i(1),  "m0(f)", self.cmdcenter.zn_get_r_i(1),  "m0_inv(f)", (self.state.zn, 1)],
                          83: [self.cmdcenter.zn_set_r_i(2),  "m0(f)", self.cmdcenter.zn_get_r_i(2),  "m0_inv(f)", (self.state.zn, 2)],
                          84: [self.cmdcenter.zn_set_r_i(3),  "m0(f)", self.cmdcenter.zn_get_r_i(3),  "m0_inv(f)", (self.state.zn, 3)],
                          85: [self.cmdcenter.zn_set_r_i(4),  "m0(f)", self.cmdcenter.zn_get_r_i(4),  "m0_inv(f)", (self.state.zn, 4)],
                          86: [self.cmdcenter.zn_set_r_i(5),  "m0(f)", self.cmdcenter.zn_get_r_i(5),  "m0_inv(f)", (self.state.zn, 5)],
                          87: [self.cmdcenter.zn_set_r_i(6),  "m1(f)", self.cmdcenter.zn_get_r_i(6),  "m1_inv(f)", (self.state.zn, 6)],
                          88: [self.cmdcenter.zn_set_r_i(7),  "m0(f)", self.cmdcenter.zn_get_r_i(7),  "m0_inv(f)", (self.state.zn, 7)]}
        self.bindings1.update(dict([(1 + i, [self.cmdcenter.zn_set_th_i(i), "m4(f)", self.cmdcenter.zn_get_th_i(i), "m4_inv(f)", (self.state.zn, i)]) for i in xrange(8)]))

        self.bindings2 = dict([(1 + i, [self.cmdcenter.par_set_i(i), "m0(f)", self.cmdcenter.par_get_i(i), "m0_inv(f)", (self.state.par, i)]) for i in xrange(8, 16)])
        self.bindings2.update(dict([(1 + i, [self.cmdcenter.zn_set_th_i(i), "m4(f)", self.cmdcenter.zn_get_th_i(i), "m4_inv(f)", (self.state.zn, i)]) for i in xrange(8, 16)]))


        # create par bindings
        self.bindings3 = dict([(81 + i, [self.cmdcenter.par_set_i(i), "m0(f)", self.cmdcenter.par_get_i(i), "m0_inv(f)", (self.state.par, i)]) for i in xrange(8)])

        self.bindings4 = dict([(81 + i, [self.cmdcenter.par_set_i(i + 8), "m0(f)", self.cmdcenter.par_get_i(i + 8), "m0_inv(f)", (self.state.par, i)]) for i in xrange(8)])

        self.bindings5 = dict([(81 + i, [self.cmdcenter.par_set_i(i + 16), "m0(f)", self.cmdcenter.par_get_i(i + 16), "m0_inv(f)", (self.state.par, i)]) for i in xrange(8)])

        self.binding_bit0 = self.binding_bit1 = 0.0


        if(hasattr(self, "midi_out")):
            self.change_bindings()

        # init thread
        threading.Thread.__init__(self)


    def change_bindings(self):
        var = self.binding_bit1 * 2 + self.binding_bit0
        if(var == 0):
            self.bindings = self.bindings1
        elif(var == 1):
            self.bindings = self.bindings2
        elif(var == 2):
            self.bindings = self.bindings3
        elif(var == 3):
            self.bindings = self.bindings1

        self.send_bindings()


    def send_bindings(self):

        # send all binding defaults
        for key in self.bindings:
            binding = self.bindings[key]
            f = binding[2]()
            f = eval(binding[3])
            self.writef(key, f)


    def writef(self, channel, f):
        val = int(f * 128.0)
        if(val == 128): val = 127
        self.midi_out.Write([[[176, channel, val, 0], pypm.Time()]])


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
            if(val == 127.0):
                f = 1.0
            else:
                f = val / 128.0

            # check bindings
            if(self.bindings.has_key(channel)):

                # process binding
                binding = self.bindings[channel]
                binding[0](eval(binding[1]))
            elif(channel == 72):
                self.binding_bit0 = f == 1.0 and 1 or 0
                self.change_bindings()
            elif(channel == 80):
                self.binding_bit1 = f == 1.0 and 1 or 0
                self.change_bindings()
