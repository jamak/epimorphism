from numpy import *
from ctypes import *
from OpenGL.GL import *
from OpenGL.GLUT import *

from phenom.keyboard import *
from phenom.mouse import *
from noumena.logger import *



class Renderer(KeyboardHandler, MouseHandler):

    def __init__(self, animator, engine):

        self.animator = animator
        self.engine = engine
        self.profile = engine.profile
        self.state = engine.state

        log("re: initializing")

        glutInit(1, [])

        # create window    
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
      
        if(self.profile.full_screen):
            log("re: fullscreen")
            glutGameModeString(str(self.profile.viewport_width) + "x" + str(self.profile.viewport_height) + ":24@" + str(self.profile.viewport_refresh))
            glutEnterGameMode()

        else:
            log("re: windowed")
            glutInitWindowSize(self.profile.viewport_width, self.profile.viewport_height)
            glutInitWindowPosition(10, 10)
            glutCreateWindow("Epimorphism")

        self.reshape(self.profile.viewport_width, self.profile.viewport_height)

        # register callbacks
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.motion)

        # generate buffer object
        size = (self.profile.kernel_dim ** 2) * 4 * sizeof(c_float)
        self.pbo = GLuint() 

        glGenBuffers(1, byref(self.pbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.pbo)     
        empty_buffer = (c_ubyte * (sizeof(c_float) * 4 * self.profile.kernel_dim ** 2))()
        glBufferData(GL_ARRAY_BUFFER, size, empty_buffer, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # register pbo with engine
        engine.register_pbo(self.pbo)

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

        # fps data
        self.d_time_start = self.d_time = self.d_timebase = glutGet(GLUT_ELAPSED_TIME) 
        self.frame_count = 0.0               

        # misc variables
        self.next_frame = False


    def set_execution(self, execution):

        glutDisplayFunc(execution)


    def cleanup(self):

        glBindBuffer(GL_ARRAY_BUFFER, self.pbo)
        glDeleteBuffers(1, self.pbo)
    

    def reshape(self, w, h):

        log("re: reshape - " + str(w) + ' ' + str(h))

        # set viewport
        self.profile.viewport_width = w
        self.profile.viewport_height = h
        self.aspect = float(w) / float(h)
        glViewport(0, 0, self.profile.viewport_width, self.profile.viewport_height)

        # configure projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)


    def do(self):      

        # compute frame rate
        self.frame_count += 1
        self.d_time = glutGet(GLUT_ELAPSED_TIME)
        if(self.frame_count % self.profile.debug_freq == 0):
            time = (1.0 * self.d_time - self.d_timebase) / self.profile.debug_freq
            avg = (1.0 * self.d_time - self.d_time_start) / self.frame_count
            print "gl time = " + str(time) + "ms"
            print "gl avg  = " + str(avg) + "ms"
            self.d_timebase = self.d_time

        # copy texture from pbo
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, self.pbo)
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

        # repost
        glutSwapBuffers()
        glutPostRedisplay()


    def start(self):

        log("re: starting")
        glutMainLoop()
