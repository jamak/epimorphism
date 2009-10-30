from common.complex import *

class EventManager(object):


    def __init__(self, cmdcenter):
        self.cmdcenter, self.state = cmdcenter, cmdcenter.state


    def handle_event(f):
        def inner(self, element, multiplier=0):
            old_time = self.state.component_switch_time
            time = 60.0 / self.state.bpm * (2 ** multiplier)
            self.state.component_switch_time = time

            f(self, element, time)

            self.state.component_switch_time = old_time

        return inner


    @handle_event
    def switch_component(self, component, multiplier=0):
        ''' Switches a component '''

        self.cmdcenter.cmd("inc_data('%s', 1)" % component)


    @handle_event
    def rotate360(self, component, time=1):
        ''' Rotates a zn by 360 deg '''

        z0 = r_to_p(self.state.zn[component])
        z1 = [z0[0], z0[1]]
        z1[1] += 2.0 * pi
        self.cmdcenter.cmd('radial_2d(zn, %d, %f, %s, %s)' % (component, time, str(z0), str(z1)))


    @handle_event
    def rotate180(self, component, time=1):
        ''' Rotates a zn by 180 deg '''

        z0 = r_to_p(self.state.zn[component])
        z1 = [z0[0], z0[1]]
        z1[1] += 2.0 * pi / 2
        self.cmdcenter.cmd('radial_2d(zn, %d, %f, %s, %s)' % (component, time, str(z0), str(z1)))


    @handle_event
    def rotate90(self, component, time=1):
        ''' Rotates a zn by 90 deg '''

        z0 = r_to_p(self.state.zn[component])
        z1 = [z0[0], z0[1]]
        z1[1] += 2.0 * pi / 4
        self.cmdcenter.cmd('radial_2d(zn, %d, %f, %s, %s)' % (component, time, str(z0), str(z1)))


    @handle_event
    def rotate45(self, component, time=1):
        ''' Rotates a zn by 45 deg '''

        z0 = r_to_p(self.state.zn[component])
        z1 = [z0[0], z0[1]]
        z1[1] += 2.0 * pi / 8
        self.cmdcenter.cmd('radial_2d(zn, %d, %f, %s, %s)' % (component, time, str(z0), str(z1)))


    @handle_event
    def rotateLoop(self, component, time=1):
        ''' Rotates a zn continuously '''

        z0 = r_to_p(self.state.zn[component])
        z1 = [z0[0], z0[1]]
        z1[1] += 2.0 * pi
        self.cmdcenter.cmd("animate_var('radial_2d', zn, %d, %s, {'s' : %s, 'e' : %s, 'loop' : True})" % (component, 16.0 * time, str(z0), str(z1)))


    @handle_event
    def transLoop(self, component, time=1):
        ''' Translates a zn continuously '''

        z0 = r_to_p(self.state.zn[component])
        z1 = [z0[0], z0[1]]
        z1[0] += 2.0
        self.cmdcenter.cmd("animate_var('radial_2d', zn, %d, %s, {'s' : %s, 'e' : %s, 'loop' : True})" % (component, 4.0 * time, str(z0), str(z1)))
