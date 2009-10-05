import sys
import logging
import logging.handlers

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s: %(lineno)d / %(funcName)s --- %(message)s")

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

fh = logging.handlers.RotatingFileHandler('log/log.txt', maxBytes=1000*100, backupCount=5)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)


def set_log(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.addHandler(fh)

    # add logger to calling context
    sys._getframe(1).f_locals['debug'] = logger.debug
