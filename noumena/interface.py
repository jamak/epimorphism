from noumena.renderer import *

from noumena.console import *
from noumena.keyboard import *
from noumena.mouse import *
from noumena.server import *
from noumena.midi import *

from common.log import *
set_log("Interface")

class Interface(object):

    def __init__(self, context):
        debug("Initializing interface")

        # set variables
        self.context = context

        self.renderer = Renderer(context)



    def sync_cmd(self, cmdcenter):

        # create input handlers
        mouse_handler = MouseHandler(cmdcenter, self.context)
        keyboard_handler = KeyboardHandler(cmdcenter)

        # create_console
        console = Console(cmdcenter)

        # register callbacks & console with Renderer
        self.renderer.register_callbacks(keyboard_handler.keyboard, mouse_handler.mouse, mouse_handler.motion)
        self.renderer.register_console_callbacks(console.render_console, console.console_keyboard)

        # start server
        if(self.context.server):
            self.server = Server(cmdcenter)
            self.server.start()

        else:
            self.server = None

        # start midi
        if(self.context.midi):
            self.midi = MidiHandler(cmdcenter)

            if(self.context.midi):
                cmdcenter.state.zn.midi = self.midi
                cmdcenter.state.par.midi = self.midi
                self.midi.start()

        else:
            self.midi = None



    def do(self):
        self.renderer.do()
