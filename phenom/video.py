import time
import os.path


class VideoRenderer(object):
    ''' The VideoRenderer object is responsible for sequentially capturing the
        frames output by the engine '''

    def __init__(self, cmdcenter, env):

        # set vars
        self.cmdcenter, self.env = cmdcenter, env
        self.frame_num = 0


    def video_time(self):

        # return the simulated video time
        return (self.frame_num * self.env.video_frame_time) / 1000.0


    def capture(self):

        # return if necessary
        if(not self.env.render_video):
            return False

        # save frame
        image = self.cmdcenter.grab_image()
        image.save("video/%s/%d.png" % (self.video_name, self.frame_num))

        # inc frame num
        self.frame_num += 1

        # stop video if necessary
        if(self.env.max_video_frames and self.frame_num == self.env.max_video_frames):
            self.stop_video(True)


    def start_video(self, video_name=None):

        # return if necessary
        if(self.contest.render_video):
            return False

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

        # set cmdcenter time function
        self.cmdcenter.time = self.video_time


    def stop_video(self, compress=False):

        # return if necessary
        if(not self.env.render_video):
            return False

        # set vars
        self.env.render_video = False

        # run script to compress video
        if(compress):
            pass

        # restore cmdcenter clock
        self.cmdcenter.time = time.clock
