from ctypes import *
from OpenGL.GL import *
from OpenGL.GLUT import *

class Console:

    def __init__(self):

        glBindTexture(GL_TEXTURE_2D, self.console_tex)  

        data = (c_ubyte * (22220 * 20 * 4 * sizeof(c_ubyte)))()

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

    
    def renderConsole(self):    

        glBindTexture(GL_TEXTURE_2D, self.console_tex)    

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(0.5, -1.0, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(0.5, -0.5, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(1.0, -0.5, 0.0)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(1.0, -1.0, 0.0)
        
        glEnd()

    def console_keyboard(self, key, x, y):
        if(key == "\033"):
            exit()

        elif(key == "`"):
            self.toggle_console()
