import time
import datetime

class Logger:
    
    echo_threshold = 3
    logging_on = True
    echo = False

    def __init__(self, file, prefix):
        self.prefix = prefix
        self.file = open('logs/' + file, 'a')

    def __del__(self):
        self.file.close        
        
    def log(self, string, level=0):
        if not self.logging_on:
            return
        now = datetime.datetime.now()
        s = '(' + now.strftime("%H:%M:%S") + ') ' + ''.join(map(lambda x: ' ', range(level))) + self.prefix + str(string)
        self.file.write(s + '\n')
        self.file.flush()
        if(self.echo and level <= self.echo_threshold):
            print s
        
