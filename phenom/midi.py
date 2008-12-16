import threading
import pypm
import re

import time

from common.complex import *


def m0(f):
    return f

def m1(f):
    return f + 1.0

def m2(f):
    return 4.0 * f + 1.0

def m3(f):
    return 5.0 * f

def m4(f):
    return 2.0 * 3.14159 * f

def m0_inv(f):
    return f

def m1_inv(f):
    return f - 1.0

def m2_inv(f):
    return (f - 1.0) / 4.0

def m3_inv(f):
    return f / 5.0

def m4_inv(f):
    return f / (2.0 * 3.14159)


class MidiHandler(threading.Thread):

    def __init__(self, cmdcenter):
        threading.Thread.__init__(self)
        self.cmdcenter = cmdcenter

        for loop in range(pypm.CountDevices()):
            interf,name,inp,outp,opened = pypm.GetDeviceInfo(loop)
            if(re.compile("BCF2000").search(name) and inp == 1):
                self.input_device = loop
                print "in", loop, name, " "
            if(re.compile("BCF2000").search(name) and outp == 1):
                self.output_device = loop
                print "out", loop, name, " "

        try:
            self.midi_in = pypm.Input(self.input_device)
            self.midi_out = pypm.Output(self.output_device, 100)
        except:
            print "MIDI device not found"
            self.cmdcenter.context.midi = False

        self.bindings1 = {81: [self.cmdcenter.zn_setter_r_i(0),  "m1(f)", self.cmdcenter.zn_getter_r(0),  "m1_inv(f)"],
                          82: [self.cmdcenter.zn_setter_r_i(1),  "m0(f)", self.cmdcenter.zn_getter_r(1),  "m0_inv(f)"],
                          83: [self.cmdcenter.zn_setter_r_i(2),  "m0(f)", self.cmdcenter.zn_getter_r(2),  "m0_inv(f)"],
                          84: [self.cmdcenter.zn_setter_r_i(3),  "m0(f)", self.cmdcenter.zn_getter_r(3),  "m0_inv(f)"],
                          85: [self.cmdcenter.zn_setter_r_i(4),  "m0(f)", self.cmdcenter.zn_getter_r(4),  "m0_inv(f)"],
                          86: [self.cmdcenter.zn_setter_r_i(5),  "m0(f)", self.cmdcenter.zn_getter_r(5),  "m0_inv(f)"],
                          87: [self.cmdcenter.zn_setter_r_i(6),  "m1(f)", self.cmdcenter.zn_getter_r(6),  "m1_inv(f)"],
                          88: [self.cmdcenter.zn_setter_r_i(7),  "m0(f)", self.cmdcenter.zn_getter_r(7),  "m0_inv(f)"],
                          1 : [self.cmdcenter.zn_setter_th_i(0), "m4(f)", self.cmdcenter.zn_getter_th(0), "m4_inv(f)"],
                          2 : [self.cmdcenter.zn_setter_th_i(1), "m4(f)", self.cmdcenter.zn_getter_th(1), "m4_inv(f)"],
                          3 : [self.cmdcenter.zn_setter_th_i(2), "m4(f)", self.cmdcenter.zn_getter_th(2), "m4_inv(f)"],
                          4 : [self.cmdcenter.zn_setter_th_i(3), "m4(f)", self.cmdcenter.zn_getter_th(3), "m4_inv(f)"],
                          5 : [self.cmdcenter.zn_setter_th_i(4), "m4(f)", self.cmdcenter.zn_getter_th(4), "m4_inv(f)"],
                          6 : [self.cmdcenter.zn_setter_th_i(5), "m4(f)", self.cmdcenter.zn_getter_th(5), "m4_inv(f)"],
                          7 : [self.cmdcenter.zn_setter_th_i(6), "m4(f)", self.cmdcenter.zn_getter_th(6), "m4_inv(f)"],
                          8 : [self.cmdcenter.zn_setter_th_i(7), "m4(f)", self.cmdcenter.zn_getter_th(7), "m4_inv(f)"]}

        self.bindings2 = dict([(81 + i, [self.cmdcenter.par_setter_i(i), "m0(f)", self.cmdcenter.par_getter_i(i), "m0_inv(f)"]) for i in xrange(8)])

        self.bindings3 = dict([(81 + i, [self.cmdcenter.par_setter_i(i + 8), "m0(f)", self.cmdcenter.par_getter_i(i + 8), "m0_inv(f)"]) for i in xrange(8)])

        self.bindings4 = dict([(81 + i, [self.cmdcenter.par_setter_i(i + 16), "m0(f)", self.cmdcenter.par_getter_i(i + 16), "m0_inv(f)"]) for i in xrange(8)])


        self.binding_bit0 = self.binding_bit1 = 0.0

        if(hasattr(self, "midi_out")):
            self.change_bindings()


    def send_bindings(self):

        for key in self.bindings:
            binding = self.bindings[key]
            print key
            f = binding[2]()
            print f
            f = eval(binding[3])
            print f
            val = int(f * 128.0)
            if(val == 128): val = 127
            print key, val
            self.midi_out.Write([[[176, key, val, 0], pypm.Time()]])


    def change_bindings(self):
        var = self.binding_bit1 * 2 + self.binding_bit0
        print var
        if(var == 0):
            self.bindings = self.bindings1
        elif(var == 1):
            self.bindings = self.bindings2
        elif(var == 2):
            self.bindings = self.bindings3
        elif(var == 3):
            self.bindings = self.bindings1

        self.send_bindings()




    def run(self):
        while(True and self.cmdcenter.context.midi):
            while not self.midi_in.Poll() and self.cmdcenter.context.midi : time.sleep(0.01)
            if(self.cmdcenter.context.midi):
                data = self.midi_in.Read(1)
                self.midi(data[0][0][1], data[0][0][2])



    def midi(self, channel, val):
        print channel, " ", val
        if(val == 127.0):
            f = 1.0
        else:
            f = val / 128.0

        if(self.bindings.has_key(channel)):
            binding = self.bindings[channel]
            binding[0](eval(binding[1]))
        elif(channel == 72):
            self.binding_bit0 = f == 1.0 and 1 or 0
            self.change_bindings()
        elif(channel == 80):
            self.binding_bit1 = f == 1.0 and 1 or 0
            self.change_bindings()


        #if(channel >= 81 and channel <= 88):
        #    th = r_to_p(self.cmdcenter.state.zn[channel - 81])[1]
        #    self.cmdcenter.state.zn[channel - 81] = p_to_r([1.0 * f + 1 + 0j, th])
        #elif(channel >= 1 and channel <= 8):
        #    mag = r_to_p(self.cmdcenter.state.zn[channel - 1])[0]
        #    self.cmdcenter.state.zn[channel - 1] = p_to_r([mag, 2 * 3.14159 * f])
