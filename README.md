# Redes-Neurais
## Documentação do Repositório de Rede Neural Básica
Este repositório contém uma implementação básica de uma rede neural utilizando a linguagem Python. A rede neural é capaz de simular uma rede com uma única camada oculta, com um número variável de neurônios em cada camada. A rede é treinada utilizando o algoritmo do Gradiente Descendente com a função de ativação Sigmoid e sua derivada.

## Arquivos
O repositório contém os seguintes arquivos:

```simular_RN.py:``` contém a implementação da rede neural, incluindo as funções de inicialização, feedforward e backpropagation.
```calcular_Erro_RN.py:``` contém a implementação do cálculo do erro entre a saída esperada e a saída produzida pela rede.
```matriz_pesos_RN.py:``` contém a implementação da matriz de pesos da rede neural, que é atualizada durante o treinamento.
```gradient_discent_RN.py:``` contém a implementação do algoritmo do Gradiente Descendente, que ajusta os pesos da rede de forma iterativa.
```treino-gradient_discent_RN.py:``` contém a implementação do treinamento da rede neural utilizando o algoritmo do Gradiente Descendente.

## Funcionamento
A rede neural é inicializada com um número variável de neurônios em cada camada, bem como com um número variável de camadas ocultas. Durante o treinamento, a matriz de pesos é atualizada utilizando o algoritmo do Gradiente Descendente, que consiste em ajustar os pesos da rede de forma iterativa, de modo a minimizar o erro entre a saída esperada e a saída produzida pela rede.

A função de ativação utilizada é a Sigmoid e sua derivada, que são calculadas durante o treinamento.

Após o treinamento, a rede neural é capaz de produzir uma saída para um conjunto de entradas dado, utilizando a função de feedforward implementada no arquivo simular_RN.py, e que foi replicada no arquivo ```treino-gradient_discent_RN.py```.

## Como usar
Para utilizar a rede neural implementada neste repositório, basta clonar o repositório para o seu computador e executar o arquivo treino-gradient_discent_RN.py. Certifique-se de que as dependências necessárias estejam instaladas.
