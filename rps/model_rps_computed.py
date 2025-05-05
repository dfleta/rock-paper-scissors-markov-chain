import torch
from pomegranate.markov_chain import MarkovChain
from file_handler import UserActionsFileHandler

# Constants
ROCK = 0
PAPER = 1
SCISSORS = 2


class RPSMarkovModel:
    def __init__(self, k=1):
        self.model = MarkovChain(k=k)
        self.X = self._initialize_training_data()
        self.train()

    def _initialize_training_data(self):
        """Initialize the model with some basic training data."""
        initial_samples = [
            [[PAPER], [SCISSORS]],
            [[PAPER], [ROCK]],
            [[PAPER], [ROCK]],
            [[PAPER], [SCISSORS]],
            [[PAPER], [ROCK]],
            [[PAPER], [PAPER]],
            [[PAPER], [ROCK]],
            [[PAPER], [ROCK]],
            [[ROCK], [ROCK]],
            [[ROCK], [ROCK]],
            [[ROCK], [PAPER]],
            [[ROCK], [SCISSORS]],
            [[ROCK], [ROCK]],
            [[ROCK], [ROCK]],
            [[ROCK], [PAPER]],
            [[ROCK], [SCISSORS]],
            [[SCISSORS], [ROCK]],
            [[SCISSORS], [PAPER]],
            [[SCISSORS], [SCISSORS]],
            [[SCISSORS], [ROCK]],
            [[SCISSORS], [SCISSORS]],
            [[SCISSORS], [ROCK]],
            [[SCISSORS], [PAPER]],
            [[SCISSORS], [SCISSORS]],
            [[SCISSORS], [ROCK]],
        ]
        return torch.tensor(initial_samples)

    @staticmethod
    def chain_to_tensor(chain):
        chain_samples = [[sample] for sample in chain]
        samples_repr_as_tensor = [
            list(pair) for pair in zip(chain_samples, chain_samples[1:])
        ]
        return torch.tensor(samples_repr_as_tensor)

    def train(self):
        for actions in UserActionsFileHandler.read_actions():
            T = self.chain_to_tensor(actions)
            self.X = torch.cat((self.X, T), 0)

        self.model.fit(self.X)

    def initial_probabilities(self):
        return self.model.distributions[0].probs[0]

    def transition_matrix(self):
        return self.model.distributions[1].probs[0]

    def series_probability(self):
        return self.model.probability(self.X)


def main() -> None:
    model = RPSMarkovModel()

    print("Initial probabilities:")
    print(model.initial_probabilities())
    print("\nTransition matrix:")
    print(model.transition_matrix())


if __name__ == "__main__":
    main()
