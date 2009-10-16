from phenom.animator import *
from phenom.setter import *
from phenom.componentmanager import *
from phenom.BM2009 import *
from common.default import *
from common.complex import *
from noumena.compiler import *
from config.configmanager import *

import StringIO
import sys

import Image

from common.log import *
set_log("CMDCENTER")

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


class CmdCenter(Setter, Animator):
    ''' The CmdCenter is the central control center for the engine and
        renderer.  All systems generating signals live here, and the object
        provides an interface for executing code int the appropriate environment. '''


    def __init__(self, state, interface, engine):
        debug("Initializing CmdCenter")

        self.state, self.interface, self.engine = state, interface, engine

        engine.frame = state

        # init componentmanager
        self.componentmanager = ComponentManager(self, self.state, self.engine)

        # init animator
        Animator.__init__(self)

        # create cmd_env function blacklist
        func_blacklist = ['do', '__del__', '__init__', 'kernel', 'print_timings', 'record_event', 'start', 'switch_kernel',
                          'keyboard', 'console_keyboard', 'register_callbacks', 'render_console', 'capture', 'render_fps',
                          'video_time', 'set_inner_loop', 'time', 'cmd', 'execute_paths', 'echo', 'reshape',
                          'set_component_indices'] + dir(object) + dir(Setter)

        # extract non-blacklist functions from an object
        def get_funcs(obj):
            return dict([(attr, getattr(obj, attr)) for attr in dir(obj) if callable(getattr(obj, attr)) and attr not in func_blacklist])

        # get functions from objects
        funcs = get_funcs(self)
        # funcs.update(get_funcs(self.renderer))
        # funcs.update(get_funcs(self.video_renderer))
        funcs.update(get_funcs(self.engine))
        funcs.update(get_funcs(self.componentmanager))
        funcs.update(default_funcs)

        # generate cmd exec environment
        self.env = CmdEnv([self.state.__dict__, self.interface.context.__dict__], funcs)        

        self.frame_cnt = 0

        # for cycling through existing states
        self.current_state_idx = -1


    def __del__(self):
        pass

    def cmd(self, code, capture=False):
        # hijack stdout, if requested
        out = StringIO.StringIO()
        sys.stdout = capture and out or sys.stdout

        err = ""

        # execute code
        if(capture):
            try:
                exec(code) in self.env
            except:
                err = traceback.format_exc().split("\n")[-2]
        else:
            exec(code) in self.env


        # restore stdout
        sys.stdout = sys.__stdout__

        # get result
        res = [out.getvalue(), err]

        # close StringIO
        out.close()

        # return result
        return res


    def do(self):
        self.frame_cnt += 1

        # execute animation paths
        self.execute_paths()


    def moduleCmd(self, module, cmd, vars):
        cmd = "self.%s.%s(**%s)" % (module, cmd, vars)
        info("Module cmd string: %s" % cmd)
        exec(cmd)


    # UTILITY FUNCTIONS

    def update_current_state_idx(self, idx):
        self.current_state_idx += idx
        self.load(self.current_state_idx)


    def t(self, val):
        self.blend_to_component("T", val)


    def load_image(self, name, buffer_name):
        ''' Loads and image into the host memory
            and uploads it to a buffer.
              buffer_name can be either fb or aux '''

        data = Image.open("image/input/" + name).convert("RGBA").tostring("raw", "RGBA", 0, -1)

        if(buffer_name == "fb"):
            self.engine.set_fb(data, True, False)
        else:
            self.engine.set_aux(data, True, False)


    def grab_image(self):
        ''' Gets the framebuffer and binds it to an Image. '''

        img = Image.frombuffer("RGBA", (self.engine.profile.kernel_dim, self.engine.profile.kernel_dim), self.engine.get_fb(), "raw", "RGBA", 0, -1).convert("RGB")

        img.show()

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
        ''' Saves the current state. '''

        name = ConfigManager().save_state(self.state, name)
        self.grab_image().save("image/image_%s.png" % name)

        info("saved state as: " % name)

        self.interface.renderer.flash_message("saved state as %s" % name)


    def load(self, name):
        ''' Loads and blends to the given state. '''

        if(isinstance(name, int)):
            name = "state_%d" % name

        info("loading state: %s" % name)

        new_state = ConfigManager().load_dict("state", name)

        updates = {}

        # get update components
        for name in self.componentmanager.component_list():
            if(getattr(self.state, name) != getattr(new_state, name)):
                updates[name] = getattr(new_state, name)

            delattr(new_state, name)

        if(not self.componentmanager.can_switch_to_components(updates)):
            error("Failed to load state")
            return False

        debug("Loading state, updating components: %s" % str(updates))

        # blend to zns
        for i in xrange(len(new_state.zn)):
            self.radial_2d(self.state.zn, i, self.state.component_switch_time, r_to_p(self.state.zn[i]), r_to_p(new_state.zn[i]))

        # blend to pars
        for i in xrange(len(new_state.par)):
            self.linear_1d(self.state.par, i, self.state.component_switch_time, self.state.par[i], new_state.par[i])

        self.componentmanager.switch_components(updates)


    def load_state(self, idx):
        ''' Loads and blends to the state with the given id. '''

        self.load("state_%d" % idx)


    def manual(self):
        ''' Toggles manual iteration. '''

        if(self.state.manual_iter):
            self.state.next_frame = True

        self.state.manual_iter = not self.state.manual_iter


    def next(self):
        ''' If manual iteration toggles, andvances frame. '''

        self.state.next_frame = True        


