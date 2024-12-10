import torch

probabilidades_iniciales = torch.tensor([0.3200, 0.3200, 0.3600])

matriz_de_transicion = torch.tensor([[0.5000, 0.2500, 0.2500],
                                     [0.6250, 0.1250, 0.2500],
                                     [0.4444, 0.2222, 0.3333]])


# pre y post action tendran que ser enum?
def transition(preaction, postaction):
    return matriz_de_transicion[preaction][postaction].item()


def max_probab_postaction_value(preaction):
    return torch.max(matriz_de_transicion[preaction]).item()


def max_probab_postaction_index(preaction):
    return torch.argmax(matriz_de_transicion[preaction]).item()

def max_initial_probability_index():
    return torch.argmax(probabilidades_iniciales).item()


if __name__ == '__main__':
    print(transition(1, 1))
    print(max_probab_postaction_value(1))
    print(max_probab_postaction_index(1))
    print(max_initial_probability_index())
