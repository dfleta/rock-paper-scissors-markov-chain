from pomegranate.markov_chain import MarkovChain
import torch

# Crear Markov chain
model = MarkovChain(k=1)

# Secuencia de observaciones
# muestras

samples = [
    [[1], [2]],  # P S
    [[1], [0]],  # P R
    [[1], [0]],  # P R
    [[1], [2]],
    [[1], [0]],
    [[1], [1]],  # P P
    [[1], [0]],
    [[1], [0]],
    [[0], [0]],
    [[0], [0]],
    [[0], [1]],
    [[0], [2]],
    [[0], [0]],
    [[0], [0]],
    [[0], [1]],
    [[0], [2]],
    [[2], [0]],
    [[2], [1]],
    [[2], [2]],
    [[2], [0]],
    [[2], [2]],
    [[2], [0]],
    [[2], [1]],
    [[2], [2]],
    [[2], [0]],
]

X = torch.tensor(samples)
model_rps = MarkovChain(k=1)
model_rps.fit(X)

print("Probabilidades iniciales:")
# Probabilidades iniciales (Categorical)
print(model_rps.distributions[0].probs[0])
# Probabilidades de transición o condicionadas (ConditionalCategorical)
print("Matriz de transicion:")
print(model_rps.distributions[1].probs[0])

print("Probabilidad de la serie:")
print(model_rps.probability(X))
