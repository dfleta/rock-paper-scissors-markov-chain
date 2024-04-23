
'''
Tutorial:
https://pomegranate.readthedocs.io/en/latest/tutorials/B_Model_Tutorial_4_Hidden_Markov_Models.html
'''

from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM
import numpy

# Modelo de observaciones para cada estado
sun = Categorical([[
    0.2, # "umbrella"
    0.8  # "no umbrella"
]])

rain = Categorical([[
    0.9,  # "umbrella"
    0.1   # "no umbrella"
]])

# Crear el modelo
model = DenseHMM()
model.add_distributions([sun, rain])

# Probabilidades de inicio
model.add_edge(model.start, sun, 0.5)
model.add_edge(model.start, rain, 0.5)

# Modelo de transiciones
model.add_edge(sun, sun, 0.8) # Prediccion de mañana si hoy = sun
model.add_edge(sun, rain, 0.2)
model.add_edge(rain, sun, 0.3) # Prediccion de mañana si hoy = rain
model.add_edge(rain, rain, 0.7)

# Datos observados / evidencia
observations = [
    "umbrella",
    "umbrella",
    "no_umbrella",
    "umbrella",
    "umbrella",
    "umbrella",
    "umbrella",
    "no_umbrella",
    "no_umbrella"
]

X = numpy.array([[[['umbrella', 'no_umbrella'].index(observation)] for observation in observations]])
print(f'Dimensiones del array de observaciones: {X.shape}')
# (1, 9, 1)

# Predecir el estado oculto, el estado del clima
y_hat = model.predict(X)

hmm_predictions = ["sun" if y.item() == 0 else "rain" for y in y_hat[0]]

for observation, prediction in zip(observations, hmm_predictions):
    print(f'{observation} -> {prediction}')

# print("observaciones:\n {}".format(' '.join(observations)))
# print("hmm pred:\n {}".format(' '.join(["sun" + "->" if y.item() == 0 else "rain" + "->" for y in y_hat[0]])))
