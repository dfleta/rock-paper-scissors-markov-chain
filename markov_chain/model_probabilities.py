from pomegranate.markov_chain import MarkovChain
import torch

# Crear Markov chain
model = MarkovChain(k=1)

# En el ejercicio de clase
# Secuencia de observaciones: R R S S R S R S R R
# 10 muestras

samples = [ [[1], [1]],
            [[1], [1]],
            [[1], [0]],
            [[0], [0]],
            [[0], [1]],
            [[1], [0]],
            [[0], [1]],
            [[1], [0]],
            [[0], [1]],
            [[1], [1]]]

X = torch.tensor(samples)
model_ejercicio = MarkovChain(k=1)
model_ejercicio.fit(X)

print("Ejercicio de clase, observaciones: R R S S R S R S R R")
print("Probabilidades iniciales:")
# Probabilidades iniciales (Categorical)
print(model_ejercicio.distributions[0].probs[0])
# Probabilidades de transición o condicionadas (ConditionalCategorical)
print("Matriz de transicion:")
print(model_ejercicio.distributions[1].probs[0])
