from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.markov_chain import MarkovChain
import torch

'''
La probabilidad de encender el robot aspirador y que se encuentre
en cada una de las habitaciones de la casa al comenzar el día es la misma.
Una vez encendido el robot, se mueve a una habitación contigua
o permanece en la misma con probabilidad proporcional
al número de la celda.
'''

# Probabilidades iniciales
start = Categorical([[
    1/6,
    1/6,
    1/6,
    1/6,
    1/6,
    1/6
]])

# Modelo de transicion
transitions = ConditionalCategorical([[
    [1/7, 2/7, 0, 4/7, 0, 0],
    [1/3, 2/3, 0, 0, 0, 0],
    [0, 0, 3/9, 0, 0, 6/9],
    [1/5, 0, 0, 4/5, 0, 0],
    [0, 0, 0, 0, 5/11, 6/11],
    [0, 0, 3/14, 0, 5/14, 6/14]
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
w = tensor([0.1127, 0.1587, 0.0913, 0.2286, 0.1353, 0.2734])
u = tensor([0.1147, 0.1380, 0.0890, 0.2473, 0.1592, 0.2518])
t = tensor([0.1118, 0.1248, 0.0836, 0.2634, 0.1623, 0.2541])
'''
