from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.markov_chain import MarkovChain
import torch

'''
Cambiamos ahora las probabilidades iniciales al encender el robot aspirador.
Las probabilidades de comenzar el día en una habitación u otra son distintas.
Una vez encendido el robot, se mueve a una habitación contigua
o permanece en la misma con igual probabilidad.
'''

# Probabilidades iniciales
start = Categorical([[
    0.2,
    0.2,
    0.1,
    0.05,
    0.05,
    0.4
]])

# Modelo de transicion
transitions = ConditionalCategorical([[
    [1/3, 1/3, 0, 1/3, 0, 0],
    [1/2, 1/2, 0, 0, 0, 0],
    [0, 0, 1/2, 0, 0, 1/2],
    [1/2, 0, 0, 1/2, 0, 0],
    [0, 0, 0, 0, 1/2, 1/2],
    [0, 0, 1/3, 0, 1/3, 1/3]
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

'''
w = tensor([0.1917, 0.1667, 0.1833, 0.0917, 0.1583, 0.2083])
u = tensor([0.1931, 0.1472, 0.1611, 0.1097, 0.1486, 0.2403])
t = tensor([0.1928, 0.1380, 0.1606, 0.1192, 0.1544, 0.2350])
'''
