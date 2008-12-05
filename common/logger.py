import time
import datetime

__file = open('common/logs/log.txt', 'a')

__echo_threshold = 3
__logging_on = True
__echo = False

def log(string, level=0):
    if not __logging_on:
        return
    now = datetime.datetime.now()
    s = '(' + now.strftime("%H:%M:%S") + ') ' + ''.join(map(lambda x: ' ', range(level))) + str(string)
    __file.write(s + '\n')
    __file.flush()
    if(__echo and level <= __echo_threshold):
        print s
        
