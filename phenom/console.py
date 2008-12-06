from ctypes import *
from OpenGL.GL import *
from OpenGL.GLUT import *

import common.glFreeType

class Console:


    FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
    #FONT_PATH = "/usr/share/fonts/truetype/aefonts/visitor1.ttf"

    def __init__(self, cmdcenter):

        self.cmdcenter, self.renderer = cmdcenter, cmdcenter.renderer

        self.console_tex = glGenTextures(1)
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

        self.console_font_size = 12
        self.max_num_status_rows = 20
        self.status_rows = []
        self.console_width = 600
        self.active_text = ""
        self.cmd_queue = []
        self.queue_idx = -1

        self.font = common.glFreeType.font_data(self.FONT_PATH, self.console_font_size)


    def render_console(self):    

        # glBindTexture(GL_TEXTURE_2D, self.console_tex)    
        glBindTexture(GL_TEXTURE_2D, 0) 

        num_rows = min(len(self.status_rows), self.max_num_status_rows)

        dims = [1.0 - 2.0 * self.console_width / self.renderer.profile.viewport_width, 
                -1.0 + 2.0 * (10 + (self.console_font_size + 4) * (1 + num_rows)) / self.renderer.profile.viewport_height]   

        dims_v = [self.renderer.profile.viewport_width - self.console_width, 0]

        # draw box
        glColor4f(0.05, 0.05, 0.1, 0.85)

        glBegin(GL_QUADS)
        glVertex3f(dims[0], dims[1], 0.0)
        glVertex3f(1.0, dims[1], 0.0)
        glVertex3f(1.0, -1.0, 0.0)
        glVertex3f(dims[0], -1.0, 0.0)
        glEnd()

        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

        # render active_text
        glColor3ub(0xff, 0xff, 0xff)
        self.font.glPrint(dims_v[0] + 6 + 5, 5, self.active_text)

        # render status
        for i in range(0, num_rows):
            row = self.status_rows[-(i + 1)]
            if(row[1] == 0):
                glColor3ub(0, 0, 0xff)
            elif(row[1] == 1):
                glColor3ub(0, 0xff, 0)
            elif(row[1] == 2):
                glColor3ub(0xff, 0, 0)
            self.font.glPrint(dims_v[0] + 6 + 5, 5 + (4 + self.console_font_size) * (i + 1), row[0])


    def console_keyboard(self, key, x, y):
        if(key == "\033"):
            exit()

        elif(key == "`"):
            self.renderer.toggle_console()

        elif(key == "\010"): # backspace
            if(self.active_text == ""):
                return 
            else:
                self.active_text = self.active_text[0:-1]
        
        elif(key == "\015"): # enter
            self.cmd_queue.append(self.active_text)
            response = self.cmdcenter.cmd(self.active_text)
            self.status_rows.append([self.active_text, 0])            
            self.active_text = ""
            self.queue_idx = 0
            for line in response[0].split("\n"):
                if(line != ""):
                    self.status_rows.append([line, 1])
            for line in response[1].split("\n"):
                if(line != ""):
                    self.status_rows.append([line, 2])

        elif(key == GLUT_KEY_UP):
            if(len(self.cmd_queue) == 0):
                return
            self.queue_idx += 1
            self.queue_idx %= len(self.cmd_queue)
            self.active_text = self.cmd_queue[-(self.queue_idx)]

        elif(key == GLUT_KEY_DOWN):
            if(len(self.cmd_queue) == 0):
                return
            self.queue_idx -= 1
            self.queue_idx %= len(self.cmd_queue)
            self.active_text = self.cmd_queue[-(self.queue_idx)]

        else:
            self.active_text += str(key)
        
            
