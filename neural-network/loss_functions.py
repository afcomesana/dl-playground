import math
import sys

class CrossEntropy:
    def __init__(self, target, output):
        
        # Prevent underflow
        output_comp = 1 - output
        
        if output == 0:
            output = sys.float_info.min
            
        if output_comp == 0:
            output_comp = sys.float_info.min

        self.loss = -(target*math.log(output) + (1 - target)*math.log(output_comp))
        self.gradient = ( (1 - target)/(output_comp) ) - (target / output)

class MeanSquaredError:
    def __init__(self, target, output):
        self.loss = 0.5*pow(output - target, 2)
        self.gradient = 2*(output - target)