from ctypes import *
from OpenGL.GL import *
from OpenGL.GLUT import *

from phenom.keyboard import *
from phenom.mouse import *

from noumena.console import *

from common.logger import *


class Renderer(Console):

    def __init__(self, profile, state):

        # set variables
        self.profile, self.state = profile, state

        log("re: initializing")

        # initialize glut
        glutInit(1, [])

        # create window    
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_ALPHA)
      
        if(self.profile.full_screen):
            log("re: fullscreen")
            glutGameModeString(str(self.profile.viewport_width) + "x" + 
                               str(self.profile.viewport_height) + ":24@" + 
                               str(self.profile.viewport_refresh))
            glutEnterGameMode()

        else:
            log("re: windowed")
            glutInitWindowSize(self.profile.viewport_width, self.profile.viewport_height)
            glutInitWindowPosition(10, 10)
            glutCreateWindow("Epimorphism")

        self.reshape(self.profile.viewport_width, self.profile.viewport_height)

        # register callbacks
        glutReshapeFunc(self.reshape)

        # generate buffer object
        size = (self.profile.kernel_dim ** 2) * 4 * sizeof(c_float)
        self.pbo = GLuint()

        glGenBuffers(1, byref(self.pbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.pbo)     
        empty_buffer = (c_float * (sizeof(c_float) * 4 * self.profile.kernel_dim ** 2))()
        glBufferData(GL_ARRAY_BUFFER, size, empty_buffer, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)



        # initialize console
        self.console_tex = glGenTextures(1)
        Console.__init__(self)

        # init gl
        glEnable(GL_TEXTURE_2D)
        glClearColor(0.0, 0.0, 0.0, 0.0)	
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST)
        glShadeModel(GL_FLAT)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # fps data
        self.d_time_start = self.d_time = self.d_timebase = glutGet(GLUT_ELAPSED_TIME) 
        self.frame_count = 0.0               

        # misc variables
        self.console = False





    def __del__(self):

        glBindBuffer(GL_ARRAY_BUFFER, self.pbo)
        glDeleteBuffers(1, self.pbo)


    def register_callbacks(self, keyboard, mouse, motion):
        self.keyboard = keyboard
        glutKeyboardFunc(keyboard)
        glutMouseFunc(mouse)
        glutMotionFunc(motion)        

    def set_inner_loop(self, inner_loop):
        glutDisplayFunc(inner_loop)
    

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


    def toggle_console(self):
        self.console = not self.console
        if(self.console):
            glutKeyboardFunc(self.console_keyboard)
        else:
            glutKeyboardFunc(self.keyboard)


    def do(self):      

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glLoadIdentity()					# Reset The View 
	# Step back (away from objects)
	glTranslatef (0.0, 0.0, -1.0)

	# Currently - NYI - No WGL text
	# Blue Text
	# glColor3ub(0, 0, 0xff)
	#
	# // Position The WGL Text On The Screen
	# glRasterPos2f(-0.40f, 0.35f);
 	# glPrint("Active WGL Bitmap Text With NeHe - %7.2f", cnt1);	

	# Red Text
	glColor3ub (0xff, 0, 0)

	glPushMatrix ()
	glLoadIdentity ()
	# Spin the text, rotation around z axe == will appears as a 2d rotation of the text on our screen
	#glRotatef (cnt1, 0, 0, 1)
	# glScalef (1, 0.8 + 0.3* cos (cnt1/5), 1)
	glTranslatef (-180, 0, 0)
	self.font.glPrint (320, 240, "Active FreeType Text - %7.2f" % (0))
	glPopMatrix ()


        # repost
        glutSwapBuffers()
        glutPostRedisplay()


    def start(self):

        log("re: starting")
        glutMainLoop()
