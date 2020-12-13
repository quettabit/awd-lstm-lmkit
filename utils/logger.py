import os

class Logger:

    def __init__(self, path):
        self.path = path
    
    def log(self, s, print_=False, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.path, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    