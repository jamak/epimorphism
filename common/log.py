import sys
import logging
import logging.handlers

# create formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s %(lineno)d %(funcName)s() - %(message)s")

# create console output
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

# create file output
fh = logging.handlers.RotatingFileHandler('log/log.txt', maxBytes=1000*100, backupCount=5)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)


def set_log(name):
    ''' Creates a loger object and adds its logging methods
        to the calling scope '''

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.addHandler(fh)

    # add logger to calling context
    sys._getframe(1).f_locals['debug']     = logger.debug
    sys._getframe(1).f_locals['info']      = logger.info
    sys._getframe(1).f_locals['warning']   = logger.warning
    sys._getframe(1).f_locals['error']     = logger.error
    sys._getframe(1).f_locals['critical']  = logger.critical
    sys._getframe(1).f_locals['exception'] = logger.exception

