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

treinamento_input = [0.0, 1.0]

resultado_esperado = [0.5, 0.1]

pesos = [3, -4, -1, 1, -3, 2, -10]
bias = [1, 2, 3, 5]

x = [-3]
y = [0.73212]

# delta_saida * peso_aresta_saida * derivada da funcao de ativacao daquele neuronio
def delta_calculate(output, a2, a3, a4, resultado, input):
    erro_output = resultado - output
    delta_output = sigmoid_derivada(output) * erro_output
    delta_a2 = delta_output * pesos[0] * sigmoid_derivada(a2)
    delta_a3 = delta_output * pesos[1] * sigmoid_derivada(a3)
    delta_a4 = delta_output * pesos[2] * sigmoid_derivada(a4)
    print("Entrada: " + str(input) + " , Resultado Esperado: " + str(resultado))
    print("saida a2 " + str(a2) + " - " + "delta a2 " + str(delta_a2))
    print("saida a3 " + str(a3) + " - " + "delta a3 " + str(delta_a3))
    print("saida a4 " + str(a4) + " - " + "delta a4 " + str(delta_a4))
    print("saida a1 (output) " + str(output) + " - " + "delta a1 (output) " + str(delta_output))
    print("\n")

def treinar_rede_neural(X, y, num_neuronios_oculta, taxa_aprendizagem, num_epocas, funcao_ativacao):
    # Inicialização dos pesos aleatórios
    pesos_camada_oculta = np.random.randn(1, num_neuronios_oculta)
    pesos_camada_saida = np.random.randn(num_neuronios_oculta, 1)

    # Loop de treinamento
    for epoca in range(num_epocas):
        # Forward pass
        for i in range(2):
            input_a5 = treinamento_input[i]
            input_a3 = sigmoid((pesos[5] * input_a5) + bias[0])
            input_a4 = sigmoid((pesos[6] * input_a5) + bias[1])    
            input_a2 = relux(((input_a3 * pesos[3]) + (input_a4 * pesos[4])) + bias[2]) 
            input_a1 = sigmoid(((input_a3 * pesos[1]) + (input_a2 * pesos[0]) + (input_a4 * pesos[2])) + bias[3]) 
            output = input_a1

        # Cálculo do erro
        erro_saida = y - output

        # Backward pass
        delta_calculate(output, input_a2, input_a3, input_a4, resultado_esperado[i], input_a5)  
        delta_saida = erro_saida * funcao_ativacao_derivada(net_saida)
        erro_camada_oculta = np.dot(delta_saida, pesos_camada_saida.T)
        delta_camada_oculta = erro_camada_oculta * funcao_ativacao_derivada(net_oculta)

        # Atualização dos pesos
        pesos_camada_saida += taxa_aprendizagem * np.dot(saida_oculta.T, delta_saida)
        pesos_camada_oculta += taxa_aprendizagem * np.dot(X.reshape(-1,1), delta_camada_oculta)

    return pesos_camada_oculta, pesos_camada_saida