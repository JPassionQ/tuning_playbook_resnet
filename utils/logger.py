import threading

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.lock = threading.Lock()

    def log(self, msg, verbose=False):
        if verbose:
            print(msg)
        with self.lock, open(self.log_path, "a") as f:
            f.write(msg + "\n")
