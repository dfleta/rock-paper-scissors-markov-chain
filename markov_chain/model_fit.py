from pomegranate.markov_chain import MarkovChain
import torch

# Crear Markov chain
model = MarkovChain(k=1)

# 50 muestras de estados de la cadena

samples = [ [[1], [1]], # rain -> rain
            [[1], [1]], # rain -> rain
            [[1], [1]], # rain -> rain
            [[0], [1]], # sun -> rain
            [[0], [1]],

        [[1],
         [1]],

        [[1],
         [1]],

        [[1],
         [1]],

        [[0],
         [0]],

        [[1],
         [0]],

        [[0],
         [0]],

        [[0],
         [0]],

        [[1],
         [0]],

        [[1],
         [1]],

        [[1],
         [1]],

        [[0],
         [1]],

        [[1],
         [0]],

        [[0],
         [1]],

        [[1],
         [0]],

        [[0],
         [1]],

        [[0],
         [0]],

        [[0],
         [0]],

        [[1],
         [1]],

        [[1],
         [0]],

        [[0],
         [0]],

        [[0],
         [1]],

        [[1],
         [1]],

        [[1],
         [0]],

        [[0],
         [0]],

        [[0],
         [0]],

        [[0],
         [1]],

        [[0],
         [0]],

        [[0],
         [0]],

        [[1],
         [0]],

        [[1],
         [0]],

        [[1],
         [0]],

        [[0],
         [0]],

        [[0],
         [0]],

        [[0],
         [0]],

        [[0],
         [0]],

        [[1],
         [1]],

        [[1],
         [1]],

        [[0],
         [0]],

        [[0],
         [1]],

        [[0],
         [0]],

        [[1],
         [1]],

        [[0],
         [0]],

        [[1],
         [0]],

        [[0],
         [0]],

        [[1],
         [1]]]

# Se las pasamos al modelo
X = torch.tensor(samples)
model.fit(X)

# Probabilidades iniciales (Categorical)
print(model.distributions[0].probs[0])

# Probabilidades de transición o condicionadas (ConditionalCategorical)
print(model.distributions[1].probs[0])
