"""
Module for reward dynamics.

TODO.

Define your reward functions here.
"""

from abc import ABC, abstractmethod


class AbstractReward(ABC):
    """
    Abstract class to define reward function.

    Make sure to implement the abstract methods in your child class.

    To implement your dopamine functionality, You will write a class \
    inheriting this abstract class. You can add attributes to your \
    child class. The dynamics of dopamine function (DA) will be \
    implemented in `compute` method. So you will call `compute` in \
    your reward-modulated learning rules to retrieve the dopamine \
    value in the desired time step. To reset or update the defined \
    attributes in your reward function, use `update` method and \
    remember to call it your learning rule computations in the \
    right place.
    """

    @abstractmethod
    def compute(self, **kwargs) -> None:
        """
        Compute the reward.

        Returns
        -------
        None
            It should return the computed reward value.

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update the internal variables.

        Returns
        -------
        None

        """
        pass


class RPE(AbstractReward):

    def __init__(
            self,
    ) -> None:
        self.predicted_reward = 0.0
        self.reward = 0.0
        self.last_reward = 0.0

    def compute(self, **kwargs) -> None:
        """
        Compute the reward.

        Returns
        -------
        None
            It should return the computed reward value.

        """

        self.predicted_reward = kwargs.get("gamma", 0.05) * self.last_reward + self.reward
        actual_reward = kwargs.get("actual_reward", 0.0)
        self.last_reward = self.reward
        self.reward = actual_reward
        return actual_reward - self.predicted_reward



    def update(self, **kwargs) -> None:
        """
        Update the internal variables.

        Returns
        -------
        None

        """
        pass


class RawReward(AbstractReward):
    def __init__(
            self,
    ) -> None:
        pass

    def compute(self, **kwargs) -> None:
        """
        Compute the reward.

        Returns
        -------
        None
            It should return the computed reward value.

        """

        actual_reward = kwargs.get("actual_reward", 0.0)

        return actual_reward



    def update(self, **kwargs) -> None:
        """
        Update the internal variables.

        Returns
        -------
        None

        """
        pass