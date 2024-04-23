from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.markov_chain import MarkovChain

# Probabilidades iniciales
start = Categorical([[
    0.5, # sun
    0.5  # rain
]])

# Modelo de transicion
transitions = ConditionalCategorical([[
    [0.8, 0.2],
    [0.3, 0.7]
]], [start])

# Crear Markov chain
model = MarkovChain([start, transitions])

# Muestrear 50 estados de la cadena
samples = model.sample(50)
'''
samples = [[[1], [1]],

        [[1],
         [1]],

        [[1],
         [1]],

        [[0],
         [1]],

        [[0],
         [1]],

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
'''

sequence = [ "sun" if sample[1][0] == 0 else "rain" for sample in samples ]
print(sequence)
