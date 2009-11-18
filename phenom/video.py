import time
import os.path

import math

from common.runner import *

from common.log import *
set_log("VIDEO")

class VideoRenderer(object):
    ''' The VideoRenderer object is responsible for sequentially capturing the
        frames output by the engine '''

    def __init__(self, cmdcenter, env):

        debug("Initializing video renderer")

        # initialize vars
        self.cmdcenter, self.env = cmdcenter, env
        self.frame_num = 0
        self.waiting_for_frame = False


    def capture(self):
        info("Capturing video frame %d at time %s" % (self.frame_num, self.cmdcenter.time()))

        # return if necessary
        if(not self.env.render_video):
            return False

        while(self.waiting_for_frame and not self.env.exit):
            time.sleep(0.01)

        # info("Done waiting")

        # define internal function for async execution
        def grab_frame():
            # info("Grabbing frame")

            # save frame
            image = self.cmdcenter.grab_image()

            # pad frame_num
            digit_size = 5
            padded = "".join(["0" for i in xrange(digit_size - int(math.log10(self.frame_num + 1)) - 1)]) + str(self.frame_num + 1)

            # save
            image.save("video/%s/%s.png" % (self.video_name, padded))

            # inc frame num
            self.frame_num += 1

            # stop video if necessary
            if(self.env.max_video_frames and self.frame_num == int(self.env.max_video_frames)):
                self.stop_video(True)

            self.waiting_for_frame = False

        # grab frame
        if(not self.env.exit):
            self.waiting_for_frame = True
            async(grab_frame)


    def start_video(self, video_name=None):

        info("Starting video renderer")

        # turn on fps sync
        self.env.fps_sync = self.env.video_frame_rate

        # set vars
        self.frame_num = 0
        self.env.render_video = True

        # get video name if necessary
        if(not video_name):
            i = 0
            while(os.path.exists("video/%d/" % i)):
                i += 1
            video_name = str(i)

        # make directory
        os.mkdir("video/%s/" % video_name)

        # set video name
        self.video_name = video_name


    def stop_video(self, compress=False):

        info("Stopping video renderer")

        # return if necessary
        if(not self.env.render_video):
            return False

        # set vars
        self.env.render_video = False

        # run script to compress video
        if(compress):
            "mencoder mf://video/2/*.png -mf w=512:h=512:fps=20:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o %s.avi" % self.video_name

        # turn off fps sync
        self.env.fps_sync = False
