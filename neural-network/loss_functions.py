import math

class CrossEntropy:
    def __init__(self, target, output):
        try:
            self.loss = -(target*math.log(output) + (1 - target)*math.log(1 - output))
            self.gradient = ( (1 - target)/(1 - output) ) - (target / output)
        except ValueError:
            print("Value error for target:",target,"and output:", output)
            self.gradient = self.loss = 0
        
class MeanSquaredError:
    def __init__(self, target, output):
        self.loss = 0.5*pow(output - target, 2)
        self.gradient = 2*(output - target)