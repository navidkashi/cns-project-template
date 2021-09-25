from abc import ABC, abstractmethod
from typing import Union, Iterable
import torch
from math import pi, sqrt
from cnsproject.utils import gaussian


class AbstractGaussianFilter(ABC):
    """

    """

    def __init__(self,
                 sigma: Union[float, Iterable[float]]
                 ) -> None:
        super(AbstractGaussianFilter, self).__init__()
        self.sigma = torch.tensor(sigma)

    @abstractmethod
    def construct(self, size) -> torch.Tensor:
        """
        Construct filter tensor.

        """
        pass


class DoG(AbstractGaussianFilter):
    def __init__(self,
                 sigma: Iterable[float],
                 on_center = True,
                 ) -> None:
        assert type(sigma) == tuple, "DoG filter sigma must be an Tuple of size 2."
        assert len(sigma) == 2, "DoG filter sigma size must be 2."
        super(DoG, self).__init__(sigma)
        self.on_center = on_center

    def construct(self, size) -> torch.Tensor:
        """
        Construct filter tensor.

        """
        line_x = torch.linspace(-int(size / 2), int(size / 2), size)
        line_y = torch.linspace(-int(size / 2), int(size / 2), size)
        x, y = torch.meshgrid(line_x, line_y)
        dog = (1./self.sigma[0]) * gaussian(torch.sqrt((torch.pow(x, 2) + torch.pow(y, 2)).float()), 0, self.sigma[0]) \
            - (1./self.sigma[1]) * gaussian(torch.sqrt((torch.pow(x, 2) + torch.pow(y, 2)).float()), 0, self.sigma[1])
        if not self.on_center:
            dog = -1 * dog
        return (1. / sqrt(2 * pi)) * dog

class Gabor(AbstractGaussianFilter):
    def __init__(self,
                 sigma: float,
                 lmbda: float,
                 theta: float,
                 gamma: float,
                 on_center = True) -> None:
        super(Gabor, self).__init__(sigma)
        self.lmbda = torch.tensor(lmbda)
        self.theta = torch.tensor(theta)
        self.gamma = torch.tensor(gamma)
        self.on_center = on_center

    def construct(self, size) -> torch.Tensor:
        """
        Construct filter tensor.

        """
        line_x = torch.linspace(-int(size / 2), int(size / 2), size)
        line_y = torch.linspace(-int(size / 2), int(size / 2), size)
        x, y = torch.meshgrid(line_x, line_y)
        rotated_x = x * torch.cos(self.theta) + y * torch.sin(self.theta)
        rotated_y = -x * torch.sin(self.theta) + y * torch.cos(self.theta)
        gabor = torch.exp(-(torch.pow(rotated_x, 2) + self.gamma ** 2 * torch.pow(rotated_y, 2)) / (2 * self.sigma ** 2)) * \
            torch.cos(2 * pi * rotated_x / self.lmbda)
        if not self.on_center:
            gabor = -1 * gabor
        return gabor
