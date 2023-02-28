import numpy as np
from random import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relux(x):
    if (x >= 0):
        return x
    else:
        return 0

pesos = [0.5, 0.4, 2, 3, 5, 8, 9]
bias = [1, 2, 3, 5]
    
def simulate():
    for i in range(10):
        input_a5 = random()
        input_a3 = sigmoid((pesos[0] * input_a5) + bias[0])
        input_a4 = sigmoid((pesos[1] * input_a5) + bias[1])    
        input_a2 = relux(((input_a3 * pesos[2]) + (input_a4 * pesos[3])) + bias[2]) 
        input_a1 = sigmoid(((input_a3 * pesos[4]) + (input_a2 * pesos[5]) + (input_a4 * pesos[6])) + bias[3]) 
        output = input_a1
        print(output)  

simulate()