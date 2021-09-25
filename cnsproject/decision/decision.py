"""
Module for decision making.

TODO.

1. Implement the dynamics of decision making. You are free to define your
   structure here.
2. Make sure to implement winner-take-all mechanism.
"""

from abc import ABC, abstractmethod
from cnsproject.network.neural_populations import NeuralPopulation
import torch

class AbstractDecision(ABC):
    """
    Abstract class to define decision making strategy.

    Make sure to implement the abstract methods in your child class.
    """

    @abstractmethod
    def compute(self, **kwargs) -> None:
        """
        Infer the decision to be made.

        Returns
        -------
        None
            It should return the decision result.

        """
        pass

    @abstractmethod
    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        pass


class WinnerTakeAllDecision(AbstractDecision):
    """
    The k-Winner-Take-All decision mechanism.

    You will have to define a constructor and specify the required \
    attributes, including k, the number of winners.
    """
    def __init__(
            self,
            **kwargs
    ) -> None:

        self.k = kwargs.get('kwta', None)
        assert self.k is not None, 'K is not defined for KWTA.'
        self.inhibition_radius = kwargs.get('inhibition_radius', None)
        assert self.inhibition_radius is not None, 'inhibition radius is not defined for KWTA.'
        self.winners = list()
        self.winner_features = torch.tensor([], dtype=torch.long)

    def compute(self, population, **kwargs) -> list:
        """
        Infer the decision to be made.

        Returns
        -------
        None
            It should return the decision result.

        """
        s = population.s.clone()
        s[self.winner_features, :] = False
        fired = torch.nonzero(s)
        k = self.k - len(self.winners)
        inhibition_potential = torch.zeros_like(population.v)
        while k > 0 and fired.shape[0] > 0:
            if k > fired.shape[0]:
                k = fired.shape[0]
            index = torch.multinomial(torch.ones(fired.shape[0]).float(), num_samples=1)
            # print(k, fired, index, fired[index][0])
            self.winners += fired[index].tolist()
            self.winner_features = torch.cat((self.winner_features, fired[index][:, 0]), 0).long()
            # print(self.winner_features)
            s[self.winner_features, :] = False
            inhibition_loc = fired[index][0][1:].tolist()
            h_min = max(0, inhibition_loc[0]-self.inhibition_radius)
            h_max = min(s.shape[1], inhibition_loc[0]+self.inhibition_radius)
            w_min = max(0, inhibition_loc[1] - self.inhibition_radius)
            w_max = min(s.shape[2], inhibition_loc[1] + self.inhibition_radius)
            inhibition_potential[:, h_min:h_max, w_min:w_max] = population.v_rest \
                                                                - population.v[:, h_min:h_max, w_min:w_max]
            fired = torch.nonzero(s)
            k = self.k - len(self.winners)
        return len(self.winners) == self.k, self.winners, inhibition_potential

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        del self.winners
        del self.winner_features
        self.winners = list()
        self.winner_features = torch.tensor([], dtype=torch.long)