from OpenGL.GL import *
from OpenGL.GLUT import *


class MouseHandler(object):
    ''' The MouseHandler is the GLUT callback that handles keyboard events
        in the Renderer object during normal opperation '''


    def __init__(self, cmdcenter, context):
        self.cmdcenter, self.context, self.state = cmdcenter, context, cmdcenter.state

        # init coords
        self.vp_start_x = 0
        self.vp_start_y = 0
        self.mouse_start_x = 0
        self.mouse_start_y = 0


    def mouse(self, button, state, x, y):
        if(state == GLUT_DOWN):
            # set start center drag coords
            if(button == 0):
                self.vp_start_x = self.context.viewport[0]
                self.vp_start_y = self.context.viewport[1]

                self.mouse_start_x = x
                self.mouse_start_y = y

        elif(state == GLUT_UP):
            # on right click, reset scale/center
            if(button == 2):
                self.context.viewport[2] = 1.0
                self.context.viewport[0] = 0.0
                self.context.viewport[1] = 0.0

            # mousewheel up, increase scale
            elif(button == 4):
                self.context.viewport[2] *= 1.1

            # mousewheel up, decrease scale
            elif(button == 3):
                self.context.viewport[2] /= 1.1


    def motion(self, x, y):
        # drag center
        self.context.viewport[0] = self.vp_start_x + self.context.viewport[2] * (x - self.mouse_start_x) / self.context.screen[0];
        self.context.viewport[1] = self.vp_start_y + self.context.viewport[2] * (y - self.mouse_start_y) / self.context.screen[1];



