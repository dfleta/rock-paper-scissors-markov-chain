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
# print(X.shape)

# Z = torch.tensor([[[0], [0]]])

file = open("user_actions_history.txt", "r", encoding="utf-8")


def chain_to_tensor(chain):
    chain_samples = [[sample] for sample in chain]
    samples_repr_as_tensor = [
        list(pair) for pair in zip(chain_samples, chain_samples[1:])
    ]
    # Los pares de transición se generan agrupando
    # las posiciones contiguas en la lista de cada
    # linea del fichero.
    # pos 1 -> pos 2; pos 2 -> pos 3...
    return torch.tensor(samples_repr_as_tensor)


for line in file:
    T = chain_to_tensor(eval(line))
    # acumulador ;)
    X = torch.cat((X, T), 0)

file.close()

model_rps = MarkovChain(k=1)
model_rps.fit(X)


def initial_probabilites():
    # Probabilidades iniciales (Categorical)
    return model_rps.distributions[0].probs[0]


def transition_matrix():
    # Probabilidades de transición o condicionadas (ConditionalCategorical)
    return model_rps.distributions[1].probs[0]


def series_probabilities():
    # Probabilidad de la serie
    return model_rps.probability(X)


if __name__ == "__main__":
    print("Probabilidades iniciales:")
    print(initial_probabilites())
    print("Matriz de transicion:")
    print(transition_matrix())
