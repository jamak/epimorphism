from logger import *
logger = Logger("log.txt", "en:  ")
log = logger.log

class Engine:
    def __init__(self, profile, state):
        self.profile, self.state = profile, state
        log("initializing")

    def start(self):
        log("starting")
