from OpenGL.GL import *
from OpenGL.GLUT import *

from phenom.cmdcenter import *

class MouseHandler:
    vp_start_x = 0
    vp_start_y = 0
    mouse_start_x = 0
    mouse_start_y = 0


    def __init__(self, cmdcenter, profile):
        self.cmdcenter, self.profile = cmdcenter, profile
        self.state = self.cmdcenter.state
        self.cmdcenter.renderer.set_mouse(self.mouse)
        self.cmdcenter.renderer.set_motion(self.motion)

    def mouse(self, button, state, x, y):
        if(state == GLUT_DOWN):
            if(button == 0):
                self.vp_start_x = self.state.vp_center_x
                self.vp_start_y = self.state.vp_center_y
                self.mouse_start_x = x
                self.mouse_start_y = y

        elif(state == GLUT_UP):
            if(button == 2):
                self.state.vp_scale = 1.0
                self.state.vp_center_x = 0.0
                self.state.vp_center_y = 0.0
            elif(button == 4):
                self.state.vp_scale *= 1.1
            elif(button == 3):
                self.state.vp_scale /= 1.1    


    def motion(self, x, y): 
        self.state.vp_center_x = self.vp_start_x + self.state.vp_scale * (x - self.mouse_start_x) / self.profile.viewport_width;
        self.state.vp_center_y = self.vp_start_y + self.state.vp_scale * (y - self.mouse_start_y) / self.profile.viewport_height;



