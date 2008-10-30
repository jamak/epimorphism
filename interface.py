from numpy import *
from ctypes import *
from OpenGL.GL import *
from OpenGL.GLUT import *

from logger import *

logger = Logger('log.txt', 'in:  ')
log = logger.log

class Interface:

    def __init__(self, engine):

        self.engine, self.profile = engine, engine.profile

        log("initializing")

        glutInit(1, [])

        # create window    
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
      
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

        # generate buffer object
        size = (self.profile.kernel_dim ** 2) * 4 * sizeof(c_float)

        pbo = GLuint() 

        glGenBuffers(1, byref(pbo))
        glBindBuffer(GL_ARRAY_BUFFER, pbo)     
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # register engine data
        self.engine.register(pbo)

        # generate texture 
        self.display_tex = GLuint()

        glGenTextures(1, byref(self.display_tex))
        glBindTexture(GL_TEXTURE_2D, self.display_tex)  

        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, self.profile.kernel_dim, self.profile.kernel_dim, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

        # init gl
        glEnable(GL_TEXTURE_2D)
        glClearColor(0.0, 0.0, 0.0, 0.0)	
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST)
        glShadeModel(GL_FLAT)
        glDisable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()    
        glMatrixMode(GL_MODELVIEW)

        # fps data
        self.d_time_start = self.d_time = self.d_timebase = glutGet(GLUT_ELAPSED_TIME) 
        self.frame_count = 0.0               

        # interface variables
        self.manual_iter = False
        self.next_frame = False
        self.vp_start_x = self.profile.vp_center_x
        self.vp_start_y = self.profile.vp_center_y
        self.mouse_start_x = 0
        self.mouse_start_y = 0

    def start(self):
        log("starting")
        self.engine.do()
        glutMainLoop()


    def display(self):      

        # compute frame rate
        self.frame_count += 1
        self.d_time = glutGet(GLUT_ELAPSED_TIME)
        if(self.frame_count % self.profile.debug_freq == 0):
            time = (1.0 * self.d_time - self.d_timebase) / self.profile.debug_freq
            avg = (1.0 * self.d_time - self.d_time_start) / self.frame_count
            print "gl time = " + str(time) + "ms"
            print "gl avg  = " + str(avg) + "ms"
            self.d_timebase = self.d_time

        # render engine
        if(not self.manual_iter or self.next_frame):
            self.next_frame = False
            self.engine.do()

        # bind texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, self.engine.pbo)
        glBindTexture(GL_TEXTURE_2D, self.display_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.profile.kernel_dim, self.profile.kernel_dim, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0)

        # compute texture coordinates
        x0 = .5 - self.profile.vp_scale / 2 - self.profile.vp_center_x * self.aspect 
        x1 = .5 + self.profile.vp_scale / 2 - self.profile.vp_center_x * self.aspect 
        y0 = .5 - self.profile.vp_scale / (2 * self.aspect) + self.profile.vp_center_y 
        y1 = .5 + self.profile.vp_scale / (2 * self.aspect) + self.profile.vp_center_y

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
        glViewport(0, 0, self.profile.viewport_width, self.profile.viewport_height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()        

    def mouse(self, button, state, x, y):
        if(state == GLUT_DOWN):
            if(button == 0):
                self.vp_start_x = self.profile.vp_center_x
                self.vp_start_y = self.profile.vp_center_y
                self.mouse_start_x = x
                self.mouse_start_y = y
        elif(state == GLUT_UP):
            if(button == 2):
                self.profile.vp_scale = 1.0
                self.profile.vp_center_x = 0.0
                self.profile.vp_center_y = 0.0
            elif(button == 4):
                self.profile.vp_scale *= 1.1
            elif(button == 3):
                self.profile.vp_scale /= 1.1    

    def motion(self, x, y):
        self.profile.vp_center_x = self.vp_start_x + self.profile.vp_scale * (x - self.mouse_start_x) / self.profile.viewport_width;
        self.profile.vp_center_y = self.vp_start_y + self.profile.vp_scale * (y - self.mouse_start_y) / self.profile.viewport_height;


    def keyboard(self, key, x, y):
        if(key == '\033'):
            self.engine.exit = True
            self.engine.cleanup()
            glBindBuffer(GL_ARRAY_BUFFER, self.engine.pbo)
            glDeleteBuffers(1, self.engine.pbo)
            exit()
        elif(key == '`'):
            if(self.manual_iter):
                self.next_frame = True
            self.manual_iter = not self.manual_iter
        elif(key == '\040'):
            self.next_frame = True
            



