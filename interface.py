from numpy import *
from ctypes import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from Image import *

from logger import *

logger = Logger('log.txt', 'in:  ')
log = logger.log

class Interface:

    def __init__(self, engine):

        self.engine, self.profile = engine, engine.profile

        log("initializing")

        glutInit(1, [])

        # create window    
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_ALPHA | GLUT_RGBA)
      
        if(self.profile.full_screen):
            log("fullscreen")

            glutGameModeString(str(self.profile.viewport_width) + "x" + str(self.profile.viewport_height) + ":24@" + str(self.profile.viewport_refresh))
            glutEnterGameMode()

        else:
            log("windowed")

            glutInitWindowSize(self.profile.viewport_width, self.profile.viewport_height)
            glutInitWindowPosition(10, 10)
            glutCreateWindow("Epimorphism")

        self.reshape(self.profile.viewport_width, self.profile.viewport_height)

        # register callbacks
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.motion)

        # generate buffer objects - code needs to be compressed 

        image = open("image189.png").tostring("raw", "RGBA", 0, -1)  

        size = (self.profile.kernel_dim ** 2) * 4 * sizeof(c_float)

        self.engine.pbo0 = GLuint() 

        glGenBuffers(1, byref(self.engine.pbo0))
        glBindBuffer(GL_ARRAY_BUFFER, self.engine.pbo0)     
        glBufferData(GL_ARRAY_BUFFER, size, image, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self.engine.register_buffer()

        # generate texture 
        self.display_tex = GLuint()

        glGenTextures(1, byref(self.display_tex))
        glBindTexture(GL_TEXTURE_2D, self.display_tex)  

        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, 1000, 1000, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

        glEnable(GL_TEXTURE_2D)
        glClearColor(0.0, 0.0, 0.0, 0.0)	

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()    
        glMatrixMode(GL_MODELVIEW)

        #fps data
        self.d_time = self.d_timebase = glutGet(GLUT_ELAPSED_TIME)
        print 2


    def start(self):
        log("starting")

        glutMainLoop()


    def display(self):      
        #self.engine.render_to_buffer()
        self.engine.do()

        self.engine.d += 1
        self.d_time = glutGet(GLUT_ELAPSED_TIME)
        if(self.d_time - self.d_timebase > 500.0):
            self.fps = self.engine.d * 1000.0 / (self.d_time - self.d_timebase)        
            print str(self.fps)
            self.d_timebase = self.d_time;		
            self.engine.d = 0        

        # print "disp = ", self.engine.d
        # print "cuda = ", self.engine.c

        # first, bind texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, self.engine.pbo0)
        glBindTexture(GL_TEXTURE_2D, self.display_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.profile.kernel_dim, self.profile.kernel_dim, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0)

        # compute texture coordinates
        x0 = .5 - self.profile.vp_scale / 2 - self.profile.vp_center_x * self.aspect 
        x1 = .5 + self.profile.vp_scale / 2 - self.profile.vp_center_x * self.aspect 
        y0 = .5 - self.profile.vp_scale / (2 * self.aspect) + self.profile.vp_center_y 
        y1 = .5 + self.profile.vp_scale / (2 * self.aspect) + self.profile.vp_center_y

        glColor(1.0,0.0,0.0)

        # render texture
        glBegin(GL_QUADS)

        glTexCoord2f(x0, y0)
        glVertex3f(-1.0, -1.0, 0)
        glTexCoord2f(x1, y0)
        glVertex3f(1.0, -1.0, 0)
        glTexCoord2f(x1, y1)
        glVertex3f(1.0, 1.0, 0)
        glTexCoord2f(x0, y1)
        glVertex3f(-1.0, 1.0, 0)

        glEnd()

        glutSwapBuffers()
        glutPostRedisplay()


    def reshape(self, w, h):
        log("reshape - " + str(w) + ' ' + str(h))

        self.profile.viewport_width = w
        self.profile.viewport_height = h
        self.aspect = float(w) / float(h)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(0, 0, self.profile.viewport_width, self.profile.viewport_height)


    def keyboard(self, key, x, y):
        if(key == '\033'):
            self.engine.exit = True
            self.engine.cleanup()
            glBindBuffer(GL_ARRAY_BUFFER, self.engine.pbo0)
            glDeleteBuffers(1, self.engine.pbo0)
            exit()


    def mouse(self, button, state, x, y):
        pass


    def motion(self, x, y):
        pass



