from ctypes import *
from OpenGL.GL import *
from OpenGL.GLUT import *

import glFreeType

class Console:

    def __init__(self):

        glBindTexture(GL_TEXTURE_2D, self.console_tex)  

        data = (c_ubyte * (20 * 20 * 4 * sizeof(c_ubyte)))()

        for i in range(0, 20*20*4):
            data[i] = 0

        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, 20, 20, 
                     0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

        self.console_font_size = 14

        self.num_status_rows = 0
        self.console_width = 500

        self.font = glFreeType.font_data("Test.ttf", 16)

    def renderConsole(self):    

        #glBindTexture(GL_TEXTURE_2D, self.console_tex)    
        glBindTexture(GL_TEXTURE_2D, 0)    


        glBegin(GL_QUADS)
        glColor4f(0.05, 0.05, 0.05, 0.7)

        dims = [1.0 - 2.0 * self.console_width / self.profile.viewport_width, 
                -1.0 + 2.0 * (20 + (self.console_font_size + 4) * (1 + self.num_status_rows)) / self.profile.viewport_height]

        glVertex3f(dims[0], dims[1], 0.0)
        glVertex3f(1.0, dims[1], 0.0)
        glVertex3f(1.0, -1.0, 0.0)
        glVertex3f(dims[0], -1.0, 0.0)
        
        glEnd()

        glColor3ub(0xff, 0, 0)

        self.font.glPrint(450, 450, "hello!!!")


    def console_keyboard(self, key, x, y):
        if(key == "\033"):
            exit()

        elif(key == "`"):
            self.toggle_console()
