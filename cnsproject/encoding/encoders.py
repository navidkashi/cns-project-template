"""
Module for encoding data into spike.
"""

from abc import ABC, abstractmethod
from typing import Optional
from cnsproject.utils import gaussian
import torch


class AbstractEncoder(ABC):
    """
    Abstract class to define encoding mechanism.

    You will define the time duration into which you want to encode the data \
    as `time` and define the time resolution as `dt`. All computations will be \
    performed on the CPU by default. To handle computation on both GPU and CPU, \
    make sure to set the device as defined in `device` attribute to all your \
    tensors. You can add any other attributes to the child classes, if needed.

    The computation procedure should be implemented in the `__call__` method. \
    Data will be passed to this method as a tensor for further computations. You \
    might need to define more parameters for this method. The `__call__`  should return \
    the tensor of spikes with the shape (time_steps, \*population.shape).

    Arguments
    ---------
    time : int
        Length of encoded tensor.
    dt : float, Optional
        Simulation time step. The default is 1.0.
    device : str, Optional
        The device to do the computations. The default is "cpu".

    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        self.time = time
        self.dt = dt
        self.device = device

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> None:
        """
        Compute the encoded tensor of the given data.

        Parameters
        ----------
        data : torch.Tensor
            The data tensor to encode.

        Returns
        -------
        None
            It should return the encoded tensor.

        """
        pass


class Time2FirstSpikeEncoder(AbstractEncoder):
    """
    Time-to-First-Spike coding.

    Implement Time-to-First-Spike coding.
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """

        Implement the computation for coding the data. Return resulting tensor.
        """
        time = int(self.time / self.dt)
        shape, size = data.shape, data.numel()
        _data = data.flatten()
        spikes = torch.zeros((time + 1, size), device=self.device)
        times = time - (_data * (time / _data.max())).long()
        spikes[times, torch.arange(size)] = 1
        spikes = spikes[:-1]
        return spikes.view(time, *shape).bool()


class PositionEncoder(AbstractEncoder):
    """
    Position coding.

    Implement Position coding.
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        sigma: float = 1.0,
        mu_step: float = 1.0,
        mu_start: float = 0.0,
        time_th: float = 0.95,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        self.sigma = sigma
        self.mu_step = mu_step
        self.mu_start = mu_start
        self.time_th = time_th
        self.size = kwargs.get("size", None)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """

        Implement the computation for coding the data. Return resulting tensor.
        """
        _data = data.flatten()
        if self.size is None:
            self.size = _data.numel()
        time = int(self.time / self.dt)
        neuron_vec = torch.zeros(self.size, device=self.device)
        times = torch.zeros((_data.numel(), neuron_vec.numel()), device=self.device)
        for i in range(_data.numel()):
            for j in range(neuron_vec.numel()):
                times[i, j] = time - gaussian(_data[i], self.mu_start + self.mu_step * j, self.sigma) * time
        times[times >= time * self.time_th] = time
        times = times.long()
        spikes = torch.zeros(time + 1, neuron_vec.numel(), device=self.device)
        spikes[times, torch.arange(neuron_vec.numel())] = 1
        spikes = spikes[:-1]
        return spikes.view(time, self.size).bool()







class PoissonEncoder(AbstractEncoder):
    """
    Poisson coding.

    Implement Poisson coding.
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        max_rate: int = 100,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        self.max_rate = max_rate

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """


        Implement the computation for coding the data. Return resulting tensor.
        """

        time = int(self.time / self.dt)

        rate = torch.zeros(*data.shape, device=self.device)
        rate[:] = ((data.view(*data.shape) / data.max()) * self.max_rate)

        rand = torch.rand(time, *data.shape, device=self.device)
        spikes = torch.where(rand <= rate * (self.dt / 1000.0), 1, 0)

        return spikes.view(time, *data.shape).bool()


class Intensity2Latency(AbstractEncoder):
    """
    Intensity2Latency coding.

    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """

        Implement the computation for coding the data. Return resulting tensor.
        """
        time = int(self.time / self.dt)
        shape, size = data.shape, data.numel()
        bins = []
        _data = data.flatten()
        nonzero_cnt = torch.nonzero(_data).size()[0]
        bin_size = nonzero_cnt // time
        sorted_data = torch.sort(_data, descending=True)
        sorted_bins_value, sorted_bins_idx = torch.split(sorted_data[0], bin_size), torch.split(
            sorted_data[1], bin_size)
        for i in range(time):
            spike_map = torch.zeros(sorted_data[0].shape)
            spike_map.scatter_(0, sorted_bins_idx[i], sorted_bins_value[i])
            spike_map = spike_map.reshape(tuple(data.shape))

            bins.append(spike_map.squeeze(0).float())
        return torch.stack(bins).bool()