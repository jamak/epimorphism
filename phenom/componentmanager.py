from phenom.datamanager import *

import time

from common.log import *
set_log("COMPONENT")


class ComponentManager(object):


    def __init__(self, cmdcenter, state):
        self.cmdcenter, self.state = cmdcenter, state

        self.switching_component = False

        # start datamanager
        self.datamanager = DataManager()

        # init indices for components
        self.component_idx = [0 for i in xrange(20)]

        self.set_component_indices()


    def set_component_indices(self):
        ''' Given the current components in state, sets the
            component index into datamanager '''

        for component_name in self.datamanager.component_names:
            idx = self.datamanager.component_names.index(component_name)
            val =  getattr(self.state, component_name.upper())

            try:
                data = [elt[0] for elt in self.datamanager.components[component_name]]
                self.component_idx[2 * idx] = data.index(val)
            except:
                error("couldn't find index for: %s - %s" %(component_name, val))
                self.component_idx[2 * idx] = 0


    def component_list(self):
        ''' Returns a list of components '''

        return self.datamanager.component_names


    def print_components(self):
        ''' Prints the currently listed components '''

        keys = self.datamanager.component_names

        # print components
        for i in xrange(len(keys)) :
            component = getattr(self.state, keys[i])
            print i+1, ":", keys[i], "-", component, "-", self.datamanager.comment(keys[i], component)


    def inc_data(self, component_name, idx):
        ''' Increments a component index '''
        debug("Inc data: %s, %s" % (component_name, idx))

        # abort if already switching
        if(self.switching_component):
            return

        # get components
        components = self.datamanager.components[component_name]

        # get and update index
        idx_idx = self.datamanager.component_names.index(component_name)

        val_idx = self.component_idx[2 * idx_idx]
        val_idx += idx
        val_idx %= len(components)

        # get component
        component = components[val_idx]

        self.switch_components({component_name: component[0]})


    def can_switch_to_components(self, data):
        ''' Checks if given components are loaded into kernel '''

        can_switch = True

        for component_name, val in data.items():
            idx_idx = self.datamanager.component_names.index(component_name)
            components = self.datamanager.components[component_name]
            try:
                component = [c for c in components if c[0] == val][0]
            except:
                warning("Can't load component: %s - %s" % (component_name, val))
                can_switch = False

        return can_switch


    def switch_components(self, data):
        ''' Switches the system to the new components specified in data '''
        debug("Switching components: %s" % str(data))

        if(len(data) == 0):
            return True

        # non-spliced
        if(not self.cmdcenter.env.splice_components):
            for component_name, val in data.items():
                idx_idx = self.datamanager.component_names.index(component_name)
                components = self.datamanager.components[component_name]
                try:
                    component = [c for c in components if c[0] == val][0]
                except:
                    error("couldn't find val in components - %s, %s" % (component_name, val))
                    return False

                val_idx = components.index(component)

                setattr(self.state, component_name, val)
                self.component_idx[2 * idx_idx] = val_idx
                self.cmdcenter.engine.compile({})
                return

        # generate updates
        updates = {}
        first_idx = None
        for component_name, val in data.items():
            idx_idx = self.datamanager.component_names.index(component_name)
            components = self.datamanager.components[component_name]
            try:
                component = [c for c in components if c[0] == val][0]
            except:
                error("couldn't find val in components - %s, %s" % (component_name, val))
                return False

            val_idx = components.index(component)
            updates[component_name] = {"val":val, "component":component, "val_idx":val_idx, "idx_idx":idx_idx}

            if(not first_idx):
                first_idx = idx_idx

            self.component_idx[2 * idx_idx + 1] = val_idx
            self.state.internal[idx_idx] = self.state.time

        # if were only changing 1 component, show a message
        if(len(updates) == 1):
            self.cmdcenter.interface.renderer.flash_message("switching %s to: %s" % (updates.keys()[0], updates[updates.keys()[0]]["val"]))

        # wait until interpolation is done
        while(self.state.time - self.state.internal[first_idx] < self.state.component_switch_time):
            time.sleep(0.1)

        # update state
        for component_name, update in updates.items():
            val_idx = update["val_idx"]
            idx_idx = update["idx_idx"]
            val = update["val"]

            setattr(self.state, component_name, val)
            self.state.internal[idx_idx] = 0
            self.component_idx[2 * idx_idx] = val_idx





