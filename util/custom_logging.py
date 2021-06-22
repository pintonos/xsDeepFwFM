import logging
import sys
import tqdm

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def get_logger(filename=None):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    if filename:
        file_handler = logging.FileHandler(filename='./logs/' + filename + '.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    #root.addHandler(TqdmLoggingHandler(level=logging.DEBUG))

    root.propagate = False

    return root