from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.markov_chain import MarkovChain
import torch

# Probabilidades iniciales
start = Categorical([[
    0.5, # sun
    0.5  # rain
]])

# Modelo de transicion
transitions = ConditionalCategorical([[
    [0.8, 0.2],
    [0.4, 0.6]
]], [start])

# Crear Markov chain
model = MarkovChain([start, transitions])

print("Probabilidades iniciales:")
# Probabilidades iniciales (Categorical)
print(model.distributions[0].probs[0])
# Probabilidades de transición o condicionadas (ConditionalCategorical)
print("Matriz de transicion:")
print(model.distributions[1].probs[0])

# vector de probabilidades iniciales
v = model.distributions[0].probs[0]

# matriz de transicion
P = model.distributions[1].probs[0]

# Probabilidades en el segundo dia de la serie
w = torch.matmul(v, P)
print(f'w = {w}')

# Probabilidades en el tercer dia de la serie
u = torch.matmul(w, P)
print(f'u = {u}')

# Probabilidades en el cuarto dia de la serie
t = torch.matmul(u, P)
print(f't = {t}')
