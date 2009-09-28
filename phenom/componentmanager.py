from phenom.datamanager import *
from noumena.compiler import *

import time

COMPILE_TIME = 1.9

class ComponentManager(object):

    def __init__(self, cmdcenter, state, renderer, engine, context):
        self.cmdcenter, self.state, self.renderer, self.engine, self.context = cmdcenter, state, renderer, engine, context

        self.switching_component = False

        # start datamanager
        self.datamanager = DataManager()

        # init indices for components
        self.set_component_indices()


    def component_list(self):
        return self.datamanager.components


    def print_components(self):
        keys = self.datamanager.components

        for i in xrange(len(keys)) :
            component = getattr(self.state, keys[i])
            print i+1, ":", keys[i], "-", component, "-", self.datamanager.comment(keys[i], component)


    def set_component_indices(self):
        self.state.component_idx = [0 for i in xrange(20)]

        component_vals = [[items[0] for items in getattr(self.datamanager, component)] for component in self.datamanager.components]

        for component_name in self.datamanager.components:
            idx = self.datamanager.components.index(component_name)
            val =  getattr(self.state, component_name.upper())

            if(component_name == "T"):
                val = val.replace("(zn[2] * z + zn[3])", "(z)").replace("zn[0] * ", "").replace(" + zn[1]", "")
            elif(component_name == "T_SEED"):
                val = val.replace("(zn[10] * z + zn[11])", "(z)").replace("zn[8] * ", "").replace(" + zn[9]", "")

            print component_name, ":", val

            self.state.component_idx[2 * idx] = component_vals[idx].index(val)


    def set_kernel(self, name):
        self.new_kernel = True
        self.engine.new_kernel = name


    def inc_data(self, component_name, idx):
        if(self.switching_component):
            return

        # get components
        components = getattr(self.datamanager, component_name)

        # get and update index
        idx_idx = self.datamanager.components.index(component_name)

        val_idx = self.state.component_idx[2 * idx_idx]
        val_idx += idx
        val_idx %= len(components)

        # get component
        component = components[val_idx]

        val = component[0]

        # switch to component
        if(not self.context.splice_components):

            # initialize component
            for line in component[1]:
                exec(line) in self.cmdcenter.env

            self.switching_component = True

            # cheat if t or t_seed
            if(component_name == "T"):
                val = "zn[0] * %s + zn[1]" % val.replace("(z)", "(zn[2] * z + zn[3])")
            elif(component_name == "T_SEED"):
                val = "zn[8] * %s + zn[9]" % val.replace("(z)", "(zn[10] * z + zn[11])")

            self.renderer.echo_string = "switching %s to: %s" % (component_name, val)

            idx_idx = self.datamanager.components.index(component_name)

            print "start switching %s" % component_name

            setattr(self.state, component_name, val)

            self.new_kernel = False
            Compiler(self.state.__dict__, self.set_kernel, self.context).start()

            while(not self.new_kernel and not self.context.exit) : time.sleep(0.1)
            if(self.context.exit) : exit()

            self.new_kernel = False

            print "done switching %s" % component_name

            self.switching_component = False

            self.set_component_indices()

            self.renderer.echo_string = None

        else:
            self.renderer.flash_message("switching %s to: %s" % (component_name, val))
            self.state.component_idx[2 * idx_idx + 1] = val_idx
            self.state.internal[idx_idx] = time.clock() - self.engine.t_start

            while(time.clock() - self.engine.t_start - self.state.internal[idx_idx] < self.context.component_switch_time):
                #print time.clock() - self.engine.t_start - self.state.internal[idx_idx], self.context.component_switch_time, time.clock(), self.engine.t_start, self.state.internal[idx_idx]
                time.sleep(0.1)
            self.state.internal[idx_idx] = 0
            self.state.component_idx[2 * idx_idx] = val_idx





