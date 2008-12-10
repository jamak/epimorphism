import threading
import pypm

class MidiHandler(threading.Thread):

    def __init__(self, cmd_center):
        threading.Thread.__init__(self)
        self.cmd_center = cmd_center

        for loop in range(pypm.CountDevices()):
            interf,name,inp,outp,opened = pypm.GetDeviceInfo(loop)
            if (inp == 1):
                print loop, name," ",
                if (inp == 1): print "(input) ",
                else: print "(output) ",
                if (opened == 1): print "(opened)"
                else: print "(unopened)"


    def run(self):
        pass
