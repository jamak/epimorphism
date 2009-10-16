from noumena.renderer import *
from noumena.video import *

from noumena.console import *
from noumena.keyboard import *
from noumena.mouse import *
from noumena.server import *
from noumena.midi import *

from common.log import *
set_log("INTERFACE")


class Interface(object):

    def __init__(self, context):
        debug("Initializing interface")

        # set variables
        self.context = context

        self.renderer = Renderer(context)


    def __del__(self):
        pass
        # kill server
        #if(self.server):
        #    self.server.__del___()


    def sync_cmd(self, cmdcenter):
        debug("Syncing with CmdCenter")

        self.cmdcenter = cmdcenter

        # create video_renderer
        self.video_renderer = VideoRenderer(self.cmdcenter, self.context)

        # start video_renderer
        if(self.context.render_video):
            self.video_renderer.video_start()

        # create input handlers
        mouse_handler = MouseHandler(self.cmdcenter, self.context)
        keyboard_handler = KeyboardHandler(self.cmdcenter, self.context)

        # create_console
        console = Console(self.cmdcenter)

        # register callbacks & console with Renderer
        self.renderer.register_callbacks(keyboard_handler.keyboard, mouse_handler.mouse, mouse_handler.motion)
        self.renderer.register_console_callbacks(console.render_console, console.console_keyboard)

        # start server
        if(self.context.server):
            self.server = Server(self.cmdcenter)
            self.server.start()

        else:
            self.server = None

        # start midi
        if(self.context.midi):
            self.midi = MidiHandler(self.cmdcenter, self.context)

            if(self.context.midi):
                # sync midi lists to controller
                self.cmdcenter.state.zn.midi = self.midi
                self.cmdcenter.state.par.midi = self.midi
                self.midi.start()

        else:
            self.midi = None


    def do(self):
        # capture video frames
        if(self.context.render_video):
            self.video_renderer.capture()

        self.renderer.do()
