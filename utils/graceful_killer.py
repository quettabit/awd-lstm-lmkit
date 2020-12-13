import signal
import time

class GracefulKiller:
    """
    https://stackoverflow.com/a/31464349/3361197
    """
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.kill_now = False
        self.kill_later = False

    def exit_gracefully(self, signum, frame):
        print("received signal {:3d}. exiting learning after current epoch .."\
                .format(signum)
        )
        self.kill_later = True

    def notify_completion(self):
        print('learning completed. moving on to testing and analysis ..')
        self.kill_now = True

    def has_kill_request(self):
        return self.kill_later
