from phenom.animator import *
from phenom.console import *
from phenom.keyboard import *
from phenom.mouse import *
from phenom.server import *
from phenom.midi import *
from phenom.video import *
from phenom.setter import *

from aer.datamanager import *

from common.default import *
from common.complex import *
from noumena.compiler import *

from config.configmanager import *

import StringIO
import sys
import time

import Image

COMPILE_TIME = 1.9


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


    def __init__(self, state, renderer, engine, context):

        self.state, self.renderer, self.engine, self.context = state, renderer, engine, context

        # start datamanager
        self.datamanager = DataManager(self.state)

        # init animator
        Animator.__init__(self)

        # create video_renderer
        self.video_renderer = VideoRenderer(self)

        # create input handlers
        mouse_handler = MouseHandler(self, renderer.profile)
        keyboard_handler = KeyboardHandler(self)

        # create_console
        console = Console(self)

        # register callbacks with Renderer
        self.renderer.register_callbacks(keyboard_handler.keyboard, mouse_handler.mouse, mouse_handler.motion,
                                         console.render_console, console.console_keyboard)

        # start server
        if(self.context.server):

            self.server = Server(self)
            self.server.start()

        else:

            self.server = None

        # start midi
        if(self.context.midi):

            self.midi = MidiHandler(self)

            if(self.context.midi):

                self.state.zn.midi = self.midi
                self.state.par.midi = self.midi
                self.midi.start()

        else:

            self.midi = None

        # start video_renderer
        if(self.context.render_video):

            self.video_renderer.video_start()

        # create cmd_env function blacklist
        func_blacklist = ['do', '__del__', '__init__', 'kernel', 'print_timings', 'record_event', 'start', 'switch_kernel',
                          'keyboard', 'console_keyboard', 'register_callbacks', 'render_console', 'capture', 'render_fps',
                          'video_time', 'set_inner_loop', 'set_new_kernel', 'time', 'cmd', 'execute_paths', 'echo', 'reshape',
                          'set_indices'] + dir(object) + dir(Setter)

        # extract non-blacklist functions from an object
        def get_funcs(obj):
            return dict([(attr, getattr(obj, attr)) for attr in dir(obj) if callable(getattr(obj, attr)) and attr not in func_blacklist])

        # get functions from objects
        funcs = get_funcs(self)
        funcs.update(get_funcs(self.renderer))
        funcs.update(get_funcs(self.video_renderer))
        funcs.update(get_funcs(self.engine))
        funcs.update(default_funcs)

        # generate cmd exec environment
        self.env = CmdEnv([self.state.__dict__, self.context.__dict__], funcs)

        # init indices for components
        self.indices = [0 for i in xrange(len(self.datamanager.__dict__))]
        self.set_indices()

        # animation settings
        self.animating = dict([(data, [None, None, None]) for data in self.datamanager.components])

        # get new kernel settings
        self.new_kernel = dict([(data, [None, None]) for data in self.datamanager.components])


    def __del__(self):

        # kill server
        if(self.server):
            self.server.__del___()


    def set_new_kernel(self, data, idx, name):

        # compiler callback
        self.new_kernel[data][idx] = name


    def set_indices(self):

        # set indices
        for component_name in self.datamanager.components:

            # get components
            components = getattr(self.datamanager, component_name)

            # get current component value
            val = getattr(self.state, component_name)

            # invert if T or T_SEED - should be replaced by surface
            if(component_name == "T"):
                val = val.replace("(zn[2] * z + zn[3])", "(z)").replace("zn[0] * ", "").replace(" + zn[1]", "")
            elif(component_name == "T_SEED"):
                val = val.replace("(zn[10] * z + zn[11])", "(z)").replace("zn[8] * ", "").replace(" + zn[9]", "")

            # get component
            component = self.datamanager.get_component_for_val(component_name, val)

            # get idx_idx
            idx_idx = self.datamanager.components.index(component_name)

            # set index
            try:
                self.indices[idx_idx] = components.index(component)
            except:
                pass


    def inc_data(self, component_name, idx):

        # get components
        components = getattr(self.datamanager, component_name)

        # get and update index
        idx_idx = self.datamanager.components.index(component_name)
        self.indices[idx_idx] += idx
        self.indices[idx_idx] %= len(components)

        # get component
        component = components[self.indices[idx_idx]]

        # initialize component
        for line in component[1]:
            exec(line) in self.env

        # switch to component
        self.blend_to_component(component_name, component[0])


    def blend_to_component(self, data, val):

        # phase 0

        idx_idx = self.datamanager.components.index(data)

        # cheat if t or t_seed
        if(data == "T"):
            val = "zn[0] * (%s) + zn[1]" % val.replace("(z)", "(zn[2] * z + zn[3])")
        elif(data == "T_SEED"):
            val = "zn[8] * (%s) + zn[9]" % val.replace("(z)", "(zn[10] * z + zn[11])")

        print "switching %s to: %s" % (data, val)
        self.renderer.echo_string = "switching %s to: %s" % (data, val)

        intrp = "((1.0f - (_clock - internal[%d]) / %ff) * (%s) + (_clock - internal[%d]) / %ff * (%s))" % (idx_idx, self.context.component_switch_time, eval("self.state." + data),
                                                                                                            idx_idx, self.context.component_switch_time, val)
        setattr(self.state, data, intrp)

        self.new_kernel[data][0] = None

        var = self.state.__dict__

        # check for multi-compile
        for key in self.datamanager.components:
            if(self.animating[key][0] and time.clock() + COMPILE_TIME < self.animating[key][0] and key != data):
                new_var = {}
                new_var.update(var)
                new_var.update({key : self.animating[key][1]})
                var = new_var
                self.animating[key][2].update({key : self.animating[key][1]})

        Compiler(var, (lambda name: self.set_new_kernel(data, 0, name)), self.context).start()

        while(not self.new_kernel[data][0] and not self.context.exit) : time.sleep(0.1)
        if(self.context.exit) : exit()

        # phase 1
        self.animating[data] = [time.clock() + self.context.component_switch_time, getattr(self.state, data), None]

        self.engine.new_kernel = self.new_kernel[data][0]

        self.new_kernel[data][1] = None

        self.state.internal[idx_idx] = time.clock() - self.engine.t_start

        setattr(self.state, data, val)

        compiler = Compiler(self.state.__dict__, (lambda name: self.set_new_kernel(data, 1, name)), self.context)

        self.animating[data][2] = compiler

        compiler.start()

        while((time.clock() < self.animating[data][0] or not self.new_kernel[data][1]) and not self.context.exit) : time.sleep(0.01)
        if(self.context.exit) : exit()

        # complete
        self.animating[data] = [None, None, None]
        self.engine.new_kernel = self.new_kernel[data][1]
        self.new_kernel[data][1] = None

        print "done switching %s" % data
        self.renderer.echo_string = None

        self.set_indices()


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

        # execute animation paths
        self.execute_paths()

        # capture video frames
        if(self.context.render_video):
            self.video_renderer.capture()


    def load_image(self, name, buffer_name):
        ''' Loads and image into the host memory
            and uploads it to a buffer.
              buffer_name can be either fb or aux '''

        data = Image.open("image/input/" + name).convert("RGBA").tostring("raw", "RGBA", 0, -1)

        if(buffer_name == "fb"):
            self.engine.set_fb(data, True)
        else:
            self.engine.set_aux(data, True)


    def grab_image(self):
        ''' Gets the framebuffer and binds it to an Image. '''

        return Image.frombuffer("RGBA", (self.engine.profile.kernel_dim, self.engine.profile.kernel_dim), self.engine.get_fb(), "raw", "RGBA", 0, -1).convert("RGB")


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

        keys = self.datamanager.components

        for i in xrange(len(keys)) :
            component = getattr(self.state, keys[i])
            print i+1, ":", keys[i], "-", component, "-", self.datamanager.comment(keys[i], component)


    def save(self, name=None):
        ''' Saves the current state. '''

        name = ConfigManager().save_state(self.state, name)
        self.grab_image().save("image/image_%s.png" % name)

        print "saved state as", name

        self.renderer.flash_message("saved state as %s" % name)


    def load(self, name):
        ''' Loads and blends to the given state. '''

        new_state = ConfigManager().load_dict(name + ".est")

        # blend to zns
        for i in xrange(len(new_state.zn)):

            self.radial_2d(self.state.zn, i, self.context.component_switch_time + COMPILE_TIME, r_to_p(self.state.zn[i]), r_to_p(new_state.zn[i]))

        # blend to pars
        for i in xrange(len(new_state.par)):
            self.linear_1d(self.state.par, i, self.context.component_switch_time + COMPILE_TIME, self.state.par[i], new_state.par[i])

        # remove zn & par from dict
        del new_state.zn
        del new_state.par

        updates = {}

        # update components
        for name in self.datamanager.components:

            if(getattr(self.state, name) != getattr(new_state, name)):

                updates[name] = getattr(new_state, name)

            delattr(new_state, name)

        # blend to components
        for data in updates:

            async(lambda : self.blend_to_component(data, updates[data]))

            time.sleep(0.2)

        # update state
        print self.state.__dict__.update(new_state.__dict__)

        # set indices
        self.set_indices()


    def load_state(self, idx):
        ''' Loads and blends to the state with the given id. '''

        self.load("state_%d" % idx)


    def manual(self):
        ''' Toggles manual iteration. '''

        if(self.context.manual_iter):
            self.context.next_frame = True
        self.context.manual_iter = not self.context.manual_iter


    def next(self):
        ''' If manual iteration toggles, andvances frame. '''

        self.context.next_frame = True



