class RealMetrics:

    def __init__(self, comet):
        self.comet = comet 
        self.mean = 0 
        self.min = float('inf')
        self.max = float('-inf')

    def reset(self):
        self.mean = 0 
        self.min = float('inf')
        self.max = float('-inf')
        

    def accumulate(self, metrics):
        if metrics['min'] < self.min:
            self.min = metrics['min']
        if metrics['max'] > self.max:
            self.max = metrics['max']
        self.mean = (self.mean + metrics['mean'])/2

    def push(self, prefix, epoch):
        self.comet.log_metrics(
            {
            'min': self.min,
            'max': self.max,
            'mean': self.mean
            },
            prefix = prefix,
            step = epoch
        )

class TopMetrics:

    def __init__(self, comet):
        self.logits = RealMetrics(comet)
        self.f_logits = RealMetrics(comet)
        self.f_logits_disabled = True

    def accumulate(self, metrics):
        self.logits.accumulate(metrics['logits'])
        if 'f_logits' in metrics:
            self.f_logits.accumulate(metrics['f_logits'])
            if self.f_logits_disabled:
                self.f_logits_disabled = False
    
    def push(self, prefix, epoch):
        self.logits.push("%s_logits" % (prefix), epoch)
        self.logits.reset()
        if not self.f_logits_disabled:
            self.f_logits.push("%s_f_logits" % (prefix), epoch)
            self.f_logits.reset()



