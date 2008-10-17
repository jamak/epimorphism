from OpenGL.GL import *
from OpenGL.GLUT import *

from logger import *

logger = Logger('log.txt', 'in:  ')
log = logger.log

class Interface:
    def __init__(self, engine, vars):
        self.engine, self.vars = engine, vars;
        log("initializing")
        glutInit(1, [])
        glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE)
        glutInitWindowSize(int(vars['viewport_width']), int(vars['viewport_height']))
        glutCreateWindow("Epimorphism")

        self.initGL()

        glutDisplayFunc(self.display)
        glutKeyboardFunc(self.keyboard)
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.motion)
        glutReshapeFunc(self.reshape)
        
    def initGL(self):

        glClearColor(0.0,0.0,0.0,1.0)
        glDisable(GL_DEPTH_TEST)

        #glViewport(0,0,window_width,window_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        #ratio = float(window_width)/float(window_height)
        #glFrustum(-1.,1.,-1.,1.,2.,10.)
        return True

    def start(self):
        log("starting")
        glutMainLoop()

    def display(self):
        pass

    def keyboard(self, key, x, y):
        
        pass

    def mouse(self, button, state, x, y):
        pass

    def motion(self, x, y):
        pass

    def reshape(self, w, h):
        log("reshape -" + str(w) + ' ' + str(h))
        self.state["viewport_width"] = w
        self.state["viewport_height"] = h
        self.aspect = float(w) / float(h)
  
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()	
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
