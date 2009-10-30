import random

from common.runner import *

from phenom.cmdcenter import *

import time

class BM2009(object):

    def __init__(self, cmdcenter):
        self.cmdcenter = cmdcenter
        self.tempo = 100.0
        self.volume = 0

        self.num_active_events = 0

        self.max_num_active_events = 4
        self.impulse_freq = 0
        random.seed()

        events0 = [["switch_seed_wt", "SEED_WT"], ["switch_seed_a", "SEED_A"]]
        events1 = [["switch_t", "T"], ["switch_t_seed", "T_SEED"], ["switch_seed_w", "SEED_W"], ["switch_seed_wt", "SEED_WT"], ["switch_seed_a", "SEED_A"]]
        events2 = []
        self.all_events = [events0, events1, events2]
        self.event_index = 0
        self.events = self.all_events[self.event_index]

        self.locked_events = {}

        self.tempo_events = []
        self.last_tempo_event_time = 0


        self.switch_exponent = 1

    def set_var(self, var, val):
        exec("self.%s = %s" % (var, val))
        print "set", var, "=", val
        exec("print self.%s" % (var))


    def switch_events(self):
        self.event_index = (self.event_index + 1) % len(self.all_events)
        self.events = self.all_events[self.event_index]
        print "switching to event list:", str(self.event_index), self.events


    def impulse(self, intensity, freq):

        # determine if we need to spawn an event
        rnd = random.random()

        spawn_threshold = intensity

        should_spawn = (self.num_active_events <= self.max_num_active_events) and (random.random() < spawn_threshold) and len(self.events) > 0

        print "should_spawn =", str(should_spawn), self.num_active_events, self.max_num_active_events, str(self.num_active_events <= self.max_num_active_events), str(random.random() < spawn_threshold)

        # choose an event
        if(should_spawn):
            rnd_idx = random.randint(0, len(self.events) - 1)
            while(self.locked_events.has_key(self.events[rnd_idx][1]) and self.locked_events[self.events[rnd_idx][1]]):
                print self.events[rnd_idx][1], "is locked"
                rnd_idx = random.randint(0, len(self.events) - 1)

            print "found event: ", str(rnd_idx), self.events[rnd_idx][0]

            self.num_active_events += 1

            print "num active events: ", self.num_active_events

            # call event
            exec("async(self.%s)" % (self.events[rnd_idx][0]), {'self':self, 'async':async})


    def spb(self):
        res = 60 / self.tempo
        print "spb:", res

        return res


    # event library

    def switch_t(self):
        self.cmdcenter.state.component_switch_time = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['T'] = True
        print "start switch_t"
        self.cmdcenter.inc_data("T", 1)
        print "stop switch_t"

        self.locked_events['T'] = False
        self.num_active_events -= 1

        self.switch_exponent = 1

    def switch_t_seed(self):
        self.cmdcenter.state.component_switch_time = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['T_SEED'] = True
        print "start switch_t_seed"
        self.cmdcenter.inc_data("T_SEED", 1)
        print "stop switch_t_seed"

        self.locked_events['T_SEED'] = False
        self.num_active_events -= 1
        self.switch_exponent = 1

    def switch_seed_w(self):
        self.cmdcenter.state.component_switch_time = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['SEED_W'] = True
        print "start switch_seed_w"
        self.cmdcenter.inc_data("SEED_W", 1)
        print "stop switch_seed_w"
        self.num_active_events -= 1
        self.locked_events['SEED_W'] = False
        self.switch_exponent = 1

    def switch_seed_wt(self):
        self.cmdcenter.state.component_switch_time = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['SEED_WT'] = True
        print "start switch_seed_wt"
        self.cmdcenter.inc_data("SEED_WT", 1)
        print "stop switch_seed_wt"
        self.num_active_events -= 1
        self.locked_events['SEED_WT'] = False
        self.switch_exponent = 1

    def switch_seed_a(self):
        self.cmdcenter.state.component_switch_time = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['SEED_A'] = True
        print "start switch_seed_a"
        self.cmdcenter.inc_data("SEED_A", 1)
        print "stop switch_seed_a"
        self.num_active_events -= 1
        self.locked_events['SEED_A'] = False
        self.switch_exponent = 1

    def switch_reduce(self):
        self.cmdcenter.state.component_switch_time = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['REDUCE'] = True


        self.cmdcenter.inc_data("REDUCE", 1)


        self.num_active_events -= 1
        self.locked_events['REDUCE'] = False
        self.switch_exponent = 1


    def path0(self):
        print "execute path"

        t = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['zn0'] = True
        print "start path0"

        z = self.cmdcenter.state.zn[0]

        print str(z), str(z.real), str(z.imag)

        print "radial_2d(zn, 0, %f, [%f, %f], [2.0, 0.0])" % (z.real, z.imag, t)

        self.cmdcenter.cmd("radial_2d(zn, 0, %f, [%f, %f], [2.0, 0.0])" % (t, z.real, z.imag))

        print "stop path0"
        self.num_active_events -= 1
        self.locked_events['zn0'] = False
        self.switch_exponent = 1


    def path1(self):
        print "execute path"

        t = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['zn1'] = True
        print "start path1"

        z = self.cmdcenter.state.zn[1]

        self.cmdcenter.cmd("radial_2d(zn, 1, %f, [%f, %f], [2.0, 0.0])" % (t, z.real, z.imag))

        print "stop path1"
        self.num_active_events -= 1
        self.locked_events['zn1'] = False
        self.switch_exponent = 1

    def path2(self):
        print "execute path"

        t = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['zn2'] = True
        print "start path2"

        z = self.cmdcenter.state.zn[2]

        self.cmdcenter.cmd("radial_2d(zn, 2, %f, [%f, %f], [2.0, 0.0])" % (t, z.real, z.imag))

        print "stop path2"
        self.num_active_events -= 1
        self.locked_events['zn2'] = False
        self.switch_exponent = 1

    def path3(self):
        print "execute path"

        t = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['zn3'] = True
        print "start path3"

        z = self.cmdcenter.state.zn[3]

        self.cmdcenter.cmd("radial_2d(zn, 3, %f, [%f, %f], [2.0, 0.0])" % (t, z.real, z.imag))

        print "stop path3"
        self.num_active_events -= 1
        self.locked_events['zn3'] = False
        self.switch_exponent = 1

    def path4(self):
        print "execute path"

        t = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['zn8'] = True
        print "start path4"

        z = self.cmdcenter.state.zn[8]

        self.cmdcenter.cmd("radial_2d(zn, 8, %f, [%f, %f], [2.0, 0.0])" % (t, z.real, z.imag))

        print "stop path4"
        self.num_active_events -= 1
        self.locked_events['zn8'] = False
        self.switch_exponent = 1

    def path5(self):
        print "execute path"

        t = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['zn9'] = True
        print "start path5"

        z = self.cmdcenter.state.zn[9]

        self.cmdcenter.cmd("radial_2d(zn, 9, %f, [%f, %f], [2.0, 0.0])" % (t, z.real, z.imag))

        print "stop path5"
        self.num_active_events -= 1
        self.locked_events['zn9'] = False
        self.switch_exponent = 1

    def path6(self):
        print "execute path"

        t = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['zn10'] = True
        print "start path6"

        z = self.cmdcenter.state.zn[10]

        self.cmdcenter.cmd("radial_2d(zn, 10, %f, [%f, %f], [2.0, 0.0])" % (t, z.real, z.imag))

        print "stop path6"
        self.num_active_events -= 1
        self.locked_events['zn10'] = False
        self.switch_exponent = 1

    def path7(self):
        print "execute path"

        t = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['zn11'] = True
        print "start path7"

        z = self.cmdcenter.state.zn[11]

        self.cmdcenter.cmd("radial_2d(zn, 11, %f, [%f, %f], [2.0, 0.0])" % (t, z.real, z.imag))

        print "stop path7"
        self.num_active_events -= 1
        self.locked_events['zn11'] = False
        self.switch_exponent = 1




    def path8(self):
        print "execute path"

        t = self.spb() * (2 ** self.switch_exponent)
        self.locked_events['zn0'] = True
        print "start path8"

        z = self.cmdcenter.state.zn[0]


        z_new = z * (0.0+1.0j)

        print "radial_2d(zn, 0, %f, [%f, %f], [%f, %f])" % (t, z.real, z.imag, z_new.real, z_new.imag)

        self.cmdcenter.cmd("radial_2d(zn, 0, %f, [%f, %f], [%f, %f])" % (t, z.real, z.imag, z_new.real, z_new.imag))

        print "stop path8"
        self.num_active_events -= 1
        self.locked_events['zn0'] = False
        self.switch_exponent = 1
