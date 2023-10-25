import math
import sys

from activation import Activation, Sigmoid

class CrossEntropy:
    
    def __init__(self, target, output, activation = Activation):
        
        offset = sys.float_info.min
        
        if issubclass(activation, Sigmoid):
            ###################################################################################
            # IMPORTANT: In this case, output should be the input of the activation function, #
            # not the output of the activation function.                                      #
            ###################################################################################
            self.loss     = target*math.log(1 + pow(math.e, -output)) + (1 - target)*(output + math.log(1 + pow(math.e, -output)))
            self.gradient = -(target/(1 + pow(math.e, -output))) + (1 - target)*(1 - 1/(1 + pow(math.e, -output)))
            
        else:
            # Normal calculation for the binary cross entropy
            self.loss     = -(target*math.log(output + offset) + (1 - target)*math.log(1 - output + offset))
            self.gradient = ( (1 - target)/(1 - output + offset) ) - (target / (output + offset))


class MeanSquaredError:
    def __init__(self, target, output):
        self.loss = 0.5*pow(output - target, 2)
        self.gradient = 2*(output - target)