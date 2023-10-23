import math

class CrossEntropy:
    def __init__(self, target, output):
        self.loss = -(target*math.log(output) + (1 - target)*math.log(1 - output))
        
        self.gradient = ( (1 - target)/(1 - output) ) - (target / output)