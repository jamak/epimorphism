from ctypes import *
from OpenGL.GL import *
from OpenGL.GLUT import *

import time

from phenom.keyboard import *
from phenom.mouse import *

from common.runner import *

import common.glFreeType
FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"


class Renderer(object):
    ''' The Renderer object is responsible for displaying the system via OpenGL/GLUT '''

    def __init__(self, state, profile, context):

        # set variables
        self.state, self.profile, self.context = state, profile, context

        # initialize glut
        glutInit(1, [])

        # create window
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)

        if(self.profile.full_screen):
            glutGameModeString(str(self.profile.viewport_width) + "x" +
                               str(self.profile.viewport_height) + ":24@" +
                               str(self.profile.viewport_refresh))
            glutEnterGameMode()

        else:
            glutInitWindowSize(self.profile.viewport_width, self.profile.viewport_height)
            glutInitWindowPosition(10, 10)
            glutCreateWindow("Epimorphism")

        # reshape
        self.reshape(self.profile.viewport_width, self.profile.viewport_height)

        # register callbacks
        glutReshapeFunc(self.reshape)

        # generate buffer object
        self.pbo = GLuint()

        glGenBuffers(1, byref(self.pbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.pbo)
        empty_buffer = (c_float * (sizeof(c_float) * 4 * self.profile.kernel_dim ** 2))()
        glBufferData(GL_ARRAY_BUFFER, (self.profile.kernel_dim ** 2) * 4 * sizeof(c_float),
                     empty_buffer, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # generate texture & set parameters
        self.display_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.display_tex)

        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, self.profile.kernel_dim, self.profile.kernel_dim,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # init gl
        glEnable(GL_TEXTURE_2D)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST)
        glShadeModel(GL_FLAT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # fps data
        self.d_time_start = self.d_time = self.d_timebase = 0
        self.frame_count = 0.0

        # misc variables
        self.show_console = False
        self.show_fps = False
        self.fps = self.fps_avg = 100
        self.fps_font_size = 16
        self.fps_font = common.glFreeType.font_data(FONT_PATH, self.fps_font_size)

        self.echo_string = None
        self.echo_font_size = int(0.0123 * self.profile.viewport_width + 2.666)
        self.echo_font = common.glFreeType.font_data(FONT_PATH, self.echo_font_size)

        self.do_main_toggle_console = False

    def __del__(self):

        # bind & delete pbo
        glBindBuffer(GL_ARRAY_BUFFER, self.pbo)
        glDeleteBuffers(1, self.pbo)


    def register_callbacks(self, keyboard, mouse, motion, render_console, console_keyboard):

        # set all standard handlers
        self.keyboard = keyboard
        glutKeyboardFunc(keyboard)
        glutSpecialFunc(keyboard)
        glutMouseFunc(mouse)
        glutMotionFunc(motion)

        # set console handers
        self.render_console = render_console
        self.console_keyboard = console_keyboard


    def render_fps(self):

        # if this isn't set font looks terrible
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

        # render text into ulc
        glColor3ub(0xff, 0xff, 0xff)
        self.fps_font.glPrint(6, self.profile.viewport_height - self.fps_font_size - 6, "fps: %.2f" % (1000.0 / self.fps))
        self.fps_font.glPrint(6, self.profile.viewport_height - 2 * self.fps_font_size - 10, "avg: %.2f" % (1000.0 / self.fps_avg))


    def echo(self):
        # if this isn't set font looks terrible
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

        # render text into llc
        glColor3ub(0xff, 0xff, 0xff)
        self.echo_font.glPrint(6, 6, self.echo_string)




    def reshape(self, w, h):

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
        if(self.context.exit) : return

        if(self.do_main_toggle_console) : self.main_toggle_console()

        # compute frame rate
        if(self.d_time == 0):
            self.frame_count = 0
            self.d_time_start = self.d_time = self.d_timebase = glutGet(GLUT_ELAPSED_TIME)
        else:
            self.frame_count += 1
            self.d_time = glutGet(GLUT_ELAPSED_TIME)
            if(self.frame_count % self.profile.debug_freq == 0):
                self.fps = (1.0 * self.d_time - self.d_timebase) / self.profile.debug_freq
                self.fps_avg = (1.0 * self.d_time - self.d_time_start) / self.frame_count
                self.d_timebase = self.d_time

        # copy texture from pbo
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, self.pbo)
        glBindTexture(GL_TEXTURE_2D, self.display_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.profile.kernel_dim, self.profile.kernel_dim,
                        GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0)

        # compute texture coordinates
        x0 = .5 - self.state.vp_scale / 2 - self.state.vp_center_x * self.aspect
        x1 = .5 + self.state.vp_scale / 2 - self.state.vp_center_x * self.aspect
        y0 = .5 - self.state.vp_scale / (2 * self.aspect) + self.state.vp_center_y
        y1 = .5 + self.state.vp_scale / (2 * self.aspect) + self.state.vp_center_y


        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

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

        # render console
        if(self.show_console):
            self.render_console()

        # render fps
        if(self.show_fps):
            self.render_fps()

        # messages
        if(self.context.echo and self.echo_string):
            self.echo()

        # repost
        glutSwapBuffers()
        glutPostRedisplay()

    def set_inner_loop(self, inner_loop):

        # set the display function to be the inner loop of the application
        glutDisplayFunc(inner_loop)


    def start(self):

        # start main glut loop
        glutMainLoop()


    def flash_message(self, msg, t=3):
        ''' This function temporarily displays a message on the screen. '''
        self.echo_string = msg

        def delayed_reset_echo():
            time.sleep(t)
            self.echo_string = None

        async(delayed_reset_echo)


    def toggle_console(self):
        ''' This function toggles the interactive console '''

        self.do_main_toggle_console = True


    # main thread callback to above function
    def main_toggle_console(self):

        self.do_main_toggle_console = False

        # toggle console
        self.show_console = not self.show_console

        # juggle keyboard handlers
        if(self.show_console):
            glutKeyboardFunc(self.console_keyboard)
            glutSpecialFunc(self.console_keyboard)
        else:
            glutSpecialFunc(self.keyboard)
            glutKeyboardFunc(self.keyboard)


    def toggle_fps(self):
        ''' This function toggles the fps display '''

        # reset debug information
        self.d_time = 0

        # toggle fps display
        self.show_fps = not self.show_fps


    def toggle_echo(self):
        ''' This function toggles the echoing '''

        # toggle echo
        self.context.echo = not self.context.echo




