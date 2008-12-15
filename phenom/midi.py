import threading
import pypm
import re

import time

from common.complex import *

class MidiHandler(threading.Thread):

    def __init__(self, cmdcenter):
        threading.Thread.__init__(self)
        self.cmdcenter = cmdcenter

        for loop in range(pypm.CountDevices()):
            interf,name,inp,outp,opened = pypm.GetDeviceInfo(loop)
            if(re.compile("BCF2000").search(name) and inp == 1):
                self.device = loop
                print loop, name, " "

        try:
            self.midi_in = pypm.Input(self.device)
        except:
            print "MIDI device not found"
            self.cmdcenter.context.midi = False


    def run(self):
        while(True and self.cmdcenter.context.midi):
            while not self.midi_in.Poll() and self.cmdcenter.context.midi : time.sleep(0.01)
            if(self.cmdcenter.context.midi):
                data = self.midi_in.Read(1)
                self.midi(data[0][0][1], data[0][0][2])



    def midi(self, channel, val):
        # print channel, " ", val
        if(val == 127.0):
            f = 1.0
        else:
            f = val / 128.0

        if(channel >= 81 and channel <= 88):
            th = r_to_p(self.cmdcenter.state.zn[channel - 81])[1]
            self.cmdcenter.state.zn[channel - 81] = p_to_r([1.0 * f + 1 + 0j, th])
        elif(channel >= 1 and channel <= 8):
            mag = r_to_p(self.cmdcenter.state.zn[channel - 1])[0]
            self.cmdcenter.state.zn[channel - 1] = p_to_r([mag, 2 * 3.14159 * f])
