from OpenGL.GL import *
from OpenGL.GLUT import *


class MouseHandler(object):
    ''' The MouseHandler is the GLUT callback that handles keyboard events
        in the Renderer object during normal opperation '''

    def __init__(self, cmdcenter, profile):
        self.cmdcenter, self.profile, self.state = cmdcenter, profile, cmdcenter.state

        # init coords
        self.vp_start_x = 0
        self.vp_start_y = 0
        self.mouse_start_x = 0
        self.mouse_start_y = 0

    def mouse(self, button, state, x, y):

        if(state == GLUT_DOWN):

            # set start center drag coords
            if(button == 0):
                self.vp_start_x = self.state.vp_center_x
                self.vp_start_y = self.state.vp_center_y

                self.mouse_start_x = x
                self.mouse_start_y = y

        elif(state == GLUT_UP):

            # on right click, reset scale/center
            if(button == 2):
                self.state.vp_scale = 1.0
                self.state.vp_center_x = 0.0
                self.state.vp_center_y = 0.0

            # mousewheel up, increase scale
            elif(button == 4):
                self.state.vp_scale *= 1.1

            # mousewheel up, decrease scale
            elif(button == 3):
                self.state.vp_scale /= 1.1


    def motion(self, x, y):

        # drag center
        self.state.vp_center_x = self.vp_start_x + self.state.vp_scale * (x - self.mouse_start_x) / self.profile.viewport_width;
        self.state.vp_center_y = self.vp_start_y + self.state.vp_scale * (y - self.mouse_start_y) / self.profile.viewport_height;



