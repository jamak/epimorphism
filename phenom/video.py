import time
import os.path

class VideoRenderer(object):

    def __init__(self, cmdcenter):

        self.cmdcenter, self.context = cmdcenter, cmdcenter.context
        self.rendering = False

        self.frame_num = 0


    def video_start(self, video_name=None):

        if(self.rendering):
            return False

        self.frame_num = 0
        self.context.render_video = self.rendering = True

        if(not video_name):
            i = 0
            while(os.path.exists("common/video/" + str(i) + "/")):
                i += 1
            video_name = str(i)

        os.mkdir("common/video/" + video_name + "/")

        self.video_name = video_name

        self.cmdcenter.time = self.video_time


    def capture(self):

        if(not self.rendering or not self.context.render_video):
            return False

        image = self.cmdcenter.grab_image()
        image.save("common/video/" + self.video_name + "/" + str(self.frame_num) + ".png")

        self.frame_num += 1

        if(self.context.max_video_frames and self.frame_num == self.context.max_video_frames):
            self.video_stop(True)


    def video_stop(self, compress=False):

        if(not self.rendering or not self.context.render_video):
            return False

        self.context.render_video = False

        if(compress): # run script to compress video
            pass

        self.rendering = False

        self.cmdcenter.time = time.clock


    def video_time(self):

        return (self.frame_num * self.context.video_frame_time) / 1000.0



