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

        # misc flags
        glEnable(GL_TEXTURE_2D)
        glClearColor(1.0, 1.0, 1.0, 1.0)

        # generate buffer objects - code needs to be compressed 
        self.engine.pbo0 = GLuint() 
        self.engine.pbo1 = GLuint()

        size = self.profile.kernel_dim * self.profile.kernel_dim * 4 * sizeof(c_float)

        glGenBuffers(1, byref(self.engine.pbo0))
        glBindBuffer(GL_ARRAY_BUFFER, self.engine.pbo0)
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW)

        glGenBuffers(1, byref(self.engine.pbo1))
        glBindBuffer(GL_ARRAY_BUFFER, self.engine.pbo0)
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW)

        self.engine.pbo_current = self.engine.pbo0

        self.engine.register_buffers()

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # generate texture 
        self.display_tex = GLuint()

        glGenTextures(1, byref(self.display_tex))
        glBindTexture(GL_TEXTURE_2D, self.display_tex)        
        
        #status = cudaGLRegisterBufferObject(vbo)
        #return vbo


    def start(self):
        log("starting")

        glutMainLoop()


    def display(self):
        # first, bind texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, self.engine.pbo_current)
        glBindTexture(GL_TEXTURE_2D, self.display_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.profile.kernel_dim, self.profile.kernel_dim, GL_BGRA, GL_UNSIGNED_BYTE, None)
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

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(0, 0, self.profile.viewport_width, self.profile.viewport_height)


    def keyboard(self, key, x, y):
        if(key == '\033'):
            self.engine.exit = True

            #glBindBuffer(1, self.engine.pbo0)
            #glDeleteBuffers(1, self.engine.pbo0)
            #status = cudaGLUnregisterBufferObject(self.engine.pbo0)

            #glBindBuffer(1, self.engine.pbo1)
            #glDeleteBuffers(1, self.engine.pbo1)
            #status = cudaGLUnregisterBufferObject(self.engine.pbo1)

            sys.exit(0)


    def mouse(self, button, state, x, y):
        pass


    def motion(self, x, y):
        pass



