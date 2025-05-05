import torch
import model_rps_computed as model
import file_handler as file_handler

# Constants for actions
ROCK = 0
PAPER = 1
SCISSORS = 2


class RPSProbabilityAnalyzer:
    """Analyzer for Rock-Paper-Scissors game probabilities."""

    def __init__(self):
        """Initialize the probability analyzer with model data."""
        self.model = model.RPSMarkovModel()
        self.initial_probabilities = self.model.initial_probabilities()
        self.transition_matrix = self.model.transition_matrix()

    def transition_probability(self, pre_action, post_action):
        """
        Get the probability of transitioning from one action to another.

        Args:
            pre_action: The initial action (ROCK=0, PAPER=1, SCISSORS=2)
            post_action: The subsequent action (ROCK=0, PAPER=1, SCISSORS=2)

        Returns:
            The transition probability
        """
        return self.transition_matrix[pre_action][post_action].item()

    def max_post_action_probability(self, pre_action):
        """
        Get the maximum probability of any action following a given action.

        Args:
            pre_action: The initial action (ROCK=0, PAPER=1, SCISSORS=2)

        Returns:
            The maximum transition probability
        """
        return torch.max(self.transition_matrix[pre_action]).item()

    def most_likely_post_action(self, pre_action):
        """
        Get the most likely action to follow a given action.

        Args:
            pre_action: The initial action (ROCK=0, PAPER=1, SCISSORS=2)

        Returns:
            The index of the most likely subsequent action
        """
        return torch.argmax(self.transition_matrix[pre_action]).item()

    def most_likely_initial_action(self):
        """
        Get the most likely initial action.

        Returns:
            The index of the most likely initial action
        """
        return torch.argmax(self.initial_probabilities).item()

    def analyze_action(self, action):
        """
        Analyze the probabilities for a given action.

        Args:
            action: The action to analyze (ROCK=0, PAPER=1, SCISSORS=2)

        Returns:
            tuple: (max probability, most likely next action,
                   probability of most likely next action)
        """
        max_prob = self.max_post_action_probability(action)
        next_action = self.most_likely_post_action(action)
        next_prob = self.transition_probability(action, next_action)
        return max_prob, next_action, next_prob


def main():
    """Main function to demonstrate the probability analyzer."""
    analyzer = RPSProbabilityAnalyzer()

    print("Paper -> Paper probability:")
    print(analyzer.transition_probability(PAPER, PAPER))

    print("\nMaximum probability after Paper:")
    print(analyzer.max_post_action_probability(PAPER))

    print("\nMost likely action after Paper:")
    print(analyzer.most_likely_post_action(PAPER))

    print("\nMost likely initial action:")
    print(analyzer.most_likely_initial_action())

    # Analyze all actions
    print("\nDetailed analysis for each action:")
    for action, name in [(ROCK, "Rock"), (PAPER, "Paper"), (SCISSORS, "Scissors")]:
        max_prob, next_action, next_prob = analyzer.analyze_action(action)
        print(f"\n{name}:")
        print(f"  Max probability: {max_prob:.3f}")
        print(f"  Most likely next: {['Rock', 'Paper', 'Scissors'][next_action]}")
        print(f"  Probability of next: {next_prob:.3f}")


if __name__ == "__main__":
    main()
