from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.markov_chain import MarkovChain
import torch

'''
Encendemos el robot asporador y lo colocamos en la habitación 1.
Una vez encendido el robot, se mueve a una habitación contigua
o permanece en la misma con igual probabilidad.
Es de esperar que las probabilidades de aspirar las habitaciones
3, 5 y 6 sean nulas en el tiempo.
¿Cómo se distribuye la probabilidad a lo largo del tiempo
en las habitaciones 1, 2 y 4?
'''

# Probabilidades iniciales
start = Categorical([[
    1.0,
    0,
    0,
    0,
    0,
    0
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
w = tensor([0.3333, 0.3333, 0.0000, 0.3333, 0.0000, 0.0000])
u = tensor([0.4444, 0.2778, 0.0000, 0.2778, 0.0000, 0.0000])
t = tensor([0.4259, 0.2870, 0.0000, 0.2870, 0.0000, 0.0000])
'''
