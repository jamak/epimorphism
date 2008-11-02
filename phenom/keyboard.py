from OpenGL.GL import *
from OpenGL.GLUT import *

class KeyboardHandler:

    def keyboard(self, key, x, y):
        if(key == "\033"):
            self.engine.exit = True
            self.engine.cleanup()
            glBindBuffer(GL_ARRAY_BUFFER, self.engine.pbo)
            glDeleteBuffers(1, self.engine.pbo)
            exit()

        elif(key == "`"):
            if(self.profile.manual_iter):
                self.next_frame = True
                self.profile.manual_iter = not self.profile.manual_iter

        elif(key == "\040"):
            self.next_frame = True
            
