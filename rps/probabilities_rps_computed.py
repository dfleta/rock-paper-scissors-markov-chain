import torch
import model_rps_computed as model

initial_probabilites = model.initial_probabilites()

transition_matrix = model.transition_matrix()

def transition_probability(preaction, postaction):
    return transition_matrix[preaction][postaction].item()


def max_probab_postaction_value(preaction):
    return torch.max(transition_matrix[preaction]).item()


def max_probab_postaction_index(preaction):
    return torch.argmax(transition_matrix[preaction]).item()


def max_initial_probability_index():
    return torch.argmax(initial_probabilites).item()


if __name__ == '__main__':
    print(transition_probability(1, 1))
    print(max_probab_postaction_value(1))
    print(max_probab_postaction_index(1))
    print(max_initial_probability_index())
