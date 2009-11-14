from phenom.video import *

from phenom.animator import *
from phenom.componentmanager import *
from phenom.script import *
from phenom.eventmanager import *
from common.default import *
from common.complex import *
from config import configmanager

import StringIO
import sys
import traceback

import Image

from common.log import *
set_log("CMDCENTER")

from random import *
from common.runner import *


class CmdEnv(dict):
    ''' The CmdEnv object is a subclass of dict used as the execution
        environment for the CmdCenter.cmd method '''

    def __init__(self, data, funcs):
        self.data, self.funcs = data, funcs


    def __getitem__(self, key):
        # first check data
        for d in self.data:
            if d.has_key(key):
                return d[key]

        # if not found, return func
        return self.funcs[key]


    def __setitem__(self, key, value):
        # set data
        for d in self.data:
            if d.has_key(key):
                d[key] = value


class CmdCenter(Animator):
    ''' The CmdCenter is the central control center for the engine and
        renderer.  All systems generating signals live here, and the object
        provides an interface for executing code int the appropriate environment. '''


    def __init__(self, env, state, interface, engine):
        debug("Initializing CmdCenter")

        self.env, self.state, self.interface, self.engine = env, state, interface, engine

        # init animator
        Animator.__init__(self)

        # init componentmanager
        self.componentmanager = ComponentManager(self, self.state)

        # init eventmanager
        self.eventmanager = EventManager(self)

        # for cycling through existing states
        self.current_state_idx = -1

        # setup application
        self.interface.renderer.set_inner_loop(self.do)
        self.frame = {}
        self.engine.frame = self.frame
        self.t_start = None
        self.t_phase = 0.0
        self.recorded_events = None

        # create cmd_env function blacklist
        func_blacklist = ['do', '__del__', '__init__', 'kernel', 'print_timings', 'record_event', 'start', 'switch_kernel',
                          'keyboard', 'console_keyboard', 'register_callbacks', 'render_console', 'capture', 'render_fps',
                          'video_time', 'set_inner_loop', 'time', 'cmd', 'execute_paths', 'echo', 'reshape',
                          'set_component_indices'] + dir(object)

        # extract non-blacklist functions & data from an object
        def get_funcs(obj):
            return dict([(attr, getattr(obj, attr)) for attr in dir(obj) if callable(getattr(obj, attr)) and attr not in func_blacklist])

        # get functions from objects
        funcs = get_funcs(self)
        funcs.update(get_funcs(self.interface.renderer))
        #funcs.update(get_funcs(self.interface.video_renderer))
        funcs.update(get_funcs(self.engine))
        funcs.update(get_funcs(self.componentmanager))
        funcs.update(default_funcs)

        # generate cmd exec environment
        self.cmd_env = CmdEnv([{"cmd":self.__dict__, "state":self.state}, self.state.__dict__, self.interface.context.__dict__, self.env.__dict__], funcs)

        # tap tempo info
        self.tempo_events = []
        self.last_tempo_event_time = 0

        self.last_event_time = 0
        seed()

        # create video_renderer
        self.video_renderer = VideoRenderer(self, self.env)

        # start video_renderer
        if(self.env.render_video):
            self.video_renderer.video_start()


    def __del__(self):
        ''' Exit handler '''

        # save events
        if(self.env.record_events and self.recorded_events):
            self.recorded_events.save()


    def start(self):
        ''' Start main loop '''
        debug("Start main loop")

        self.t_start = time.clock()
        self.state.frame_cnt = 0

        self.interface.renderer.start()


    def do(self):
        ''' Main application loop '''

        # execute engine
        if((not (self.env.manual_iter and not self.env.next_frame)) and not self.env.freeze):
            self.env.next_frame = False

            # get time
            if(self.env.fps_sync):
                self.state.time = self.state.frame_cnt / self.env.fps_sync + self.t_phase
            else:
                self.state.time = time.clock() - self.t_start + self.t_phase


            #print str(self.state.time), str(self.t_phase)


            # execute animation paths
            self.execute_paths()

            # render frame
            self.send_frame()
            self.engine.do()

            self.state.frame_cnt += 1

        # execute interface
        self.interface.do()

        # capture video frames
        if(self.env.render_video):
            self.video_renderer.capture()

        # cleanup
        if(self.env.exit):
            self.interface.renderer.stop()


        #if(self.time() - self.last_event_time > 10):
        #    print "EVENT!!!"
        #    self.last_event_time = self.time()

            #i = randint(0,2)
#            if(i == 0):
 #               async(lambda :self.componentmanager.inc_data('T', 1))
  #          elif(i == 1):
   #             async(lambda :self.componentmanager.inc_data('SEED_W', 1))





    def send_frame(self):
        ''' Generates and sends the current frame to the Engine '''

        clock = self.time()
        data = {"type":"float", "val":clock}
        self.frame["_clock"] = data

        data = {"type":"float_array", "val":self.state.par}
        self.frame["par"] = data

        data = {"type":"float_array", "val":self.state.internal}
        self.frame["internal"] = data

        data = {"type":"int_array", "val":self.componentmanager.component_idx}
        self.frame["component_idx"] = data

        data = {"type":"complex_array", "val":self.state.zn}
        self.frame["zn"] = data

        data = {"type":"float", "val":self.state.component_switch_time}
        self.frame["switch_time"] = data


    def cmd(self, code, capture=False):
        ''' Execute code in the CmdEnv environment '''

        if(self.env.record_events):
            if(not self.recorded_events): self.recorded_events = Script(self)
            self.recorded_events.push(self.time(), code)

        # debug("Executing cmd: %s", code)

        # hijack stdout, if requested
        out = StringIO.StringIO()
        sys.stdout = capture and out or sys.stdout

        err = ""

        # execute code
        if(capture):
            try:
                exec(code) in self.cmd_env
            except:
                err = traceback.format_exc().split("\n")[-2]
        else:
            exec(code) in self.cmd_env


        # restore stdout
        sys.stdout = sys.__stdout__

        # get result
        res = [out.getvalue(), err]

        # close StringIO
        out.close()

        # return result
        return res


    # UTILITY FUNCTIONS


    def set_val(self, val, var, idx):
        self.cmd("%s[%s] = %s" % (var, (((type(idx) == int) and "%s" or "'%s'") % idx), val))


    def get_val(self, var, idx):
        return eval("self.%s[%s]" % (var, (((type(idx) == int) and "%s" or "'%s'") % idx)))


    def time(self):
        ''' Returns current system time '''

        return self.state.time


    def update_current_state_idx(self, idx):
        self.current_state_idx += idx
        self.load(self.current_state_idx)


    def load_image(self, name, buffer_name):
        ''' Loads and image into the host memory
            and uploads it to a buffer.
              buffer_name can be either fb or aux '''
        debug("Load image: %s", name)

        data = Image.open("image/input/" + name).convert("RGBA").tostring("raw", "RGBA", 0, -1)

        if(buffer_name == "fb"):
            self.engine.set_fb(data, True, False)

        else:
            self.engine.set_aux(data, True, False)


    def grab_image(self):
        ''' Gets the framebuffer and binds it to an Image. '''
        debug("Grab image")

        img = Image.frombuffer("RGBA", (self.engine.profile.kernel_dim, self.engine.profile.kernel_dim),
                               self.engine.get_fb(), "raw", "RGBA", 0, -1).convert("RGB")

        # img.show()
        return img


    def pars(self):
        ''' Prints a list of paramaters, their bindings, and their values. '''

        for i in xrange(len(self.state.par_names)):
            print self.state.par_names[i], ":", i


    def funcs(self):
        ''' Prints a list of all functions available in the command environment. '''

        # sort keys
        keys = self.env.funcs.keys()
        keys.sort()

        for key in keys : print key


    def components(self):
        ''' Prints a list of all components, their bindings, and their values. '''

        self.componentmanager.print_components()


    def save(self, name=None):
        ''' Grabs a screenshot and saves the current state. '''

        name = configmanager.outp_obj("state", self.state.__dict__, name)
        info("saved state as: %s" % name)

        img = self.grab_image()

        self.env.freeze = True
        async(lambda : self.__save_image(img, name))
        # img.show()


    def __save_image(self, img, name):
        ''' Save image '''
        img.save("image/image_%s.png" % name)
        self.interface.renderer.flash_message("saved state as %s" % name)
        self.env.freeze = False


    def load(self, name):
        ''' Loads and blends to the given state. '''

        if(isinstance(name, int)):
            name = "state_%d" % name

        info("Loading state: %s" % name)

        new_state = configmanager.load_dict("state", name)

        updates = {}

        # get update components
        for name in self.componentmanager.component_list():
            if(self.state.components[name] != new_state.components[name]):
                updates[name] = new_state.components[name]

            del(new_state.components[name])

        if(not self.componentmanager.can_switch_to_components(updates)):
            error("Failed to load state")
            return False

        debug("Loading state, updating components: %s" % str(updates))

        # blend to zns
        for i in xrange(len(new_state.zn)):
            self.cmd('radial_2d(zn, %d, component_switch_time, %s, %s)' % (i, str(r_to_p(self.state.zn[i])), str(r_to_p(new_state.zn[i]))))

        # blend to pars
        for i in xrange(len(new_state.par)):
            self.cmd('linear_1d(par, %d, component_switch_time, %f, %f)' % (i, self.state.par[i], new_state.par[i]))


        # print new_state.time, self.state.time, self.t_phase, self.state.component_switch_time

        # shift t_start
        # self.cmd('linear_1d(cmd, "t_phase", component_switch_time, %f, %f)' % (0, state.time + self.t_phase - 1.0) / 1.0))

        self.cmd("switch_components(%s)" % str(updates))


    def load_state(self, idx):
        ''' Loads and blends to the state with the given id. '''

        self.load("state_%d" % idx)


    def manual(self):
        ''' Toggles manual iteration. '''

        if(self.env.manual_iter):
            self.env.next_frame = True

        self.env.manual_iter = not self.env.manual_iter


    def next(self):
        ''' If manual iteration toggles, andvances frame. '''

        self.env.next_frame = True


    def tap_tempo(self):
        ''' Uses tap tempo to set bmp '''

        t = self.time()

        # reset if necessary
        if(t - self.last_tempo_event_time > 2):
            self.tempo_events = []

        # set & append
        self.last_tempo_event_time = t
        self.tempo_events.append(t)

        # max 20 events
        if(len(self.tempo_events) > 20):
            self.tempo_events.pop(0)

        # compute tempo
        if(len(self.tempo_events) > 1):
            lst = [self.tempo_events[i + 1] - self.tempo_events[i] for i in xrange(len(self.tempo_events) - 1)]
            self.state.bmp = 1.0 / (sum(lst) / (len(self.tempo_events) - 1)) * 60
            info("Tempo: %s bmp" % self.state.bmp)
