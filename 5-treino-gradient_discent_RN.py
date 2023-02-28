import numpy as np
from random import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    return x * (1-x)

def relux(x):
    if (x >= 0):
        return x
    else:
        return 0

num_epocas = 5

pesos = [3, -4, -1, 1, -3, 2, -10]
bias = [1, 2, 3, 5]

taxa_aprendizagem = .1

x = [-3, -2.0,-1.0,-0.5,0.0 ,0.5 ,1.0 ,1.5 ,2.0 ,3.0 ]
y = [0.73212, 0.7339, 0.7838, 0.8903, 0.9820, 0.8114, 0.5937, 0.5219, 0.5049, 0.5002]



def treinar_rede_neural():

    # Loop de treinamento
    for epoca in range(num_epocas):
        # Forward pass
        for i in range(10):
            input_a5 = x[i]
            input_a3 = sigmoid((pesos[5] * input_a5) + bias[0])
            input_a4 = sigmoid((pesos[6] * input_a5) + bias[1])    
            input_a2 = relux(((input_a3 * pesos[3]) + (input_a4 * pesos[4])) + bias[2]) 
            input_a1 = sigmoid(((input_a3 * pesos[1]) + (input_a2 * pesos[0]) + (input_a4 * pesos[2])) + bias[3]) 
            output = input_a1
            

            # Backward pass
            erro_output = y[i] - output
            delta_output = sigmoid_derivada(output) * erro_output
            delta_a2 = delta_output * pesos[0] * sigmoid_derivada(input_a2)
            delta_a3 = delta_output * pesos[1] * sigmoid_derivada(input_a3)
            delta_a4 = delta_output * pesos[2] * sigmoid_derivada(input_a4)

            # Atualização dos pesos
            pesos[0] = pesos[0] - (delta_output * taxa_aprendizagem) 
            pesos[1] = pesos[1] - (delta_output * taxa_aprendizagem) 
            pesos[2] = pesos[2] - (delta_output * taxa_aprendizagem) 
            pesos[3] = pesos[3] - (delta_a2 * taxa_aprendizagem) 
            pesos[4] = pesos[4] - (delta_a2 * taxa_aprendizagem) 
            pesos[5] = pesos[5] - (delta_a3 * taxa_aprendizagem) 
            pesos[6] = pesos[6] - (delta_a4 * taxa_aprendizagem) 

        print("Pesos : ", pesos)
        print("\n")


treinar_rede_neural()