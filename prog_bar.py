import sys
#import time

class progress_bar:
    def __init__(self, iterations , steps = 5):
        self.n = iterations
        self.steps = steps
        self.w = int(float(self.n)/float(self.steps))
        self.overage = int(self.n/self.steps)
        
    def set(self):
        self.ind = 0
        sys.stdout.write('[' + ' '*self.w + ']')
        sys.stdout.flush()
        sys.stdout.write('\b' * (self.w+1) )
        sys.stdout.flush()

    def step(self):
       self.ind += 1
       if self.ind % self.steps == 0:     
           sys.stdout.write('=')
           sys.stdout.flush()

       else:
           pass

       if self.ind == self.n:
           sys.stdout.write(']')
           sys.stdout.flush()
           sys.stdout.write('\n')
           sys.stdout.flush()

class percent_bar:
    def __init__(self, iterations , digits = 6, steps = 5):
        self.n = iterations
        self.w = digits+3
        self.steps = steps
        self.overage = int(self.n/self.steps)
        
    def set(self):
        self.ind = 0
        self.perc = str( round(100*float(self.ind) / float(self.n),1) )        
        sys.stdout.write('[' + self.perc + '%'  + ']')
        sys.stdout.flush()
        sys.stdout.write('\b' * (len(self.perc)+2) )
        sys.stdout.flush()

    def step(self):
        self.ind += 1

        if self.ind % self.steps == 0:
            self.perc = str( round(100*float(self.ind) / float(self.n),1) )        
            sys.stdout.write(self.perc + '%'  + ']')
            sys.stdout.flush()
            sys.stdout.write('\b' * (len(self.perc)+2) )
            sys.stdout.flush()

        else:
            pass


        if self.ind == self.n:
            sys.stdout.write('\b')
            sys.stdout.flush()
