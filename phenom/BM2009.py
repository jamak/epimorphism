import random

from common.runner import *

from phenom.cmdcenter import *

class BM2009(object):

    def __init__(self, cmdcenter):
        self.cmdcenter = cmdcenter
        self.tempo = 0
        self.volume = 0

        self.num_active_events = 0

        self.max_num_active_events = 4
        self.impulse_freq = 0
        random.seed()

        self.events = [["switch_t", 1.0], ["switch_t_seed", 1], ["switch_seed_w", 1], ["switch_seed_wt", 1], ["switch_seed_w", 1], ["switch_seed_a", 1], ["switch_reduce", 1]]


    def impulse(self, intensity, freq):

        local_intensity = (intensity + self.volume) / 2

        print "local_intensity =" + str(local_intensity)

        # determine if we need to spawn an event
        rnd = random.random()
        should_spawn = (self.num_active_events <= self.max_num_active_events) and (random.random() < local_intensity)

        print "should_spawn =", str(should_spawn), self.num_active_events, self.max_num_active_events, str(self.num_active_events <= self.max_num_active_events), str(random.random() < local_intensity)

        # choose an event
        if(should_spawn):
            rnd_idx = random.randint(0, len(self.events) - 1)
            while(self.events[rnd_idx][1] < local_intensity):
                rnd_idx = random.randint(0, len(self.events) - 1)

            print "found event: ", str(rnd_idx), self.events[rnd_idx][0]

            self.num_active_events += 1

            print "num active events: ", self.num_active_events

            # call event
            exec("async(self.%s)" % (self.events[rnd_idx][0]), {'self':self, 'async':async})

    def mspb(self):
        res = 60 * 1000 / self.tempo
        print "mbps:", res

        return 60 * 1000 / self.tempo


    def set_var(self, var, val):
        exec("self.%s = %s" % (var, val))
        print "set", var, "=", val
        exec("print self.%s" % (var))


    # event library

    def switch_t(self):
        #self.cmdcenter.context.component_switch_time = (self.mspb() / 1000.0) * (2 ** random.randint(0, 2))
        print "start switch_t"
        self.cmdcenter.inc_data("T", 1)
        print "stop switch_t"

        self.num_active_events -= 1

    def switch_t_seed(self):
        #self.cmdcenter.context.component_switch_time = (self.mspb() / 1000.0) * (2 ** random.randint(0, 2))
        print "start switch_t_seed"
        self.cmdcenter.inc_data("T_SEED", 1)
        print "stop switch_t_seed"

        self.num_active_events -= 1

    def switch_seed_w(self):
        #self.cmdcenter.context.component_switch_time = (self.mspb() / 1000.0) * (2 ** random.randint(0, 2))
        print "start switch_seed_w"
        self.cmdcenter.inc_data("SEED_W", 1)
        print "stop switch_seed_w"
        self.num_active_events -= 1

    def switch_seed_wt(self):
        #self.cmdcenter.context.component_switch_time = (self.mspb() / 1000.0) * (2 ** random.randint(0, 2))
        print "start switch_seed_w"
        self.cmdcenter.inc_data("SEED_WT", 1)
        print "stop switch_seed_w"
        self.num_active_events -= 1

    def switch_seed_a(self):
        #self.cmdcenter.context.component_switch_time = (self.mspb() / 1000.0) * (2 ** random.randint(0, 2))
        print "start switch_seed_w"
        self.cmdcenter.inc_data("SEED_A", 1)
        print "stop switch_seed_w"
        self.num_active_events -= 1

    def switch_reduce(self):
        #self.cmdcenter.context.component_switch_time = (self.mspb() / 1000.0) * (2 ** random.randint(0, 2))
        print "start switch_seed_w"
        self.cmdcenter.inc_data("REDUCE", 1)
        print "stop switch_seed_w"
        self.num_active_events -= 1
