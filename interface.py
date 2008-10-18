from OpenGL.GL import *
from OpenGL.GLUT import *

from logger import *

logger = Logger('log.txt', 'in:  ')
log = logger.log

class Interface:
    def __init__(self, engine, profile):
        self.engine, self.profile = engine, profile;
        log("initializing")
        glutInit(1, [])
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_ALPHA | GLUT_RGBA);

        if(profile.full_screen):
            log("fullscreen")
            glutGameModeString(str(profile.viewport_width) + "x" + str(profile.viewport_height) + ":24@" + str(profile.viewport_refresh))
            glutEnterGameMode()
        else:
            log("windowed")
            glutInitWindowSize(profile.viewport_width, profile.viewport_height)
            glutInitWindowPosition(10, 10);
            glutCreateWindow("Epimorphism")

        glutDisplayFunc(self.display)
        glutIdleFunc(self.display)
        glutKeyboardFunc(self.keyboard)
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.motion)
        glutReshapeFunc(self.reshape)

        glEnable(GL_TEXTURE_2D)
        glClearColor(1.0, 1.0, 1.0, 1.0)

        self.reshape(self.profile.viewport_width, self.profile.viewport_height)


    def start(self):
        log("starting")
        glutMainLoop()


    def display(self):
        x0, x1, y0, y1 = .5 - self.profile.vp_scale / 2 - self.profile.vp_center_x * self.aspect, .5 + self.profile.vp_scale / 2 - self.profile.vp_center_x * self.aspect, .5 - self.profile.vp_scale / (2 * self.aspect) + self.profile.vp_center_y, .5 + self.profile.vp_scale / (2 * self.aspect) + self.profile.vp_center_y
        glBegin(GL_QUADS)
        glVertex3f(-1.0, -1.0, 0)
        glVertex3f(1.0, -1.0, 0)
        glVertex3f(1.0, 1.0, 0)
        glVertex3f(-1.0, 1.0, 0)
        glEnd()
        glutSwapBuffers();


    def keyboard(self, key, x, y):
        if(key == '\033'):
            sys.exit(0)


    def mouse(self, button, state, x, y):
        pass


    def motion(self, x, y):
        pass


    def reshape(self, w, h):
        log("reshape -" + str(w) + ' ' + str(h))
        self.profile.viewport_width = w
        self.profile.viewport_height = h
        self.aspect = float(w) / float(h)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(0, 0, self.profile.viewport_width, self.profile.viewport_height)


