"""
Module for learning rules.
"""

from abc import ABC
from typing import Union, Optional, Sequence

import numpy as np
import torch
from cnsproject.utils import sign
from ..network.connections import *


class LearningRule(ABC):
    """
    Abstract class for defining learning rules.

    Each learning rule will be applied on a synaptic connection defined as \
    `connection` attribute. It possesses learning rate `lr` and weight \
    decay rate `weight_decay`. You might need to define more parameters/\
    attributes to the child classes.

    Implement the dynamics in `update` method of the classes. Computations \
    for weight decay and clamping the weights has been implemented in the \
    parent class `update` method. So do not invent the wheel again and call \
    it at the end  of the child method.

    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.

    """

    def __init__(
            self,
            connection: AbstractConnection,
            lr: Optional[Union[float, Sequence[float]]] = None,
            weight_decay: float = 0.,
            **kwargs
    ) -> None:
        self.connection = connection
        if lr is None:
            lr = [0., 0.]
        elif isinstance(lr, float) or isinstance(lr, int):
            lr = [lr, lr]

        self.lr = torch.tensor(lr, dtype=torch.float)

        self.weight_decay = 1 - weight_decay if weight_decay else 1.

    def update(self) -> None:
        """
        Abstract method for a learning rule update.

        Returns
        -------
        None

        """
        if self.weight_decay:
            self.connection.w *= self.weight_decay

        if (
                self.connection.wmin != -np.inf or self.connection.wmax != np.inf
        ) and not isinstance(self.connection, NoOp):
            self.connection.w.clamp_(self.connection.wmin,
                                     self.connection.wmax)


class NoOp(LearningRule):
    """
    Learning rule with no effect.

    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.

    """

    def __init__(
            self,
            connection: AbstractConnection,
            lr: Optional[Union[float, Sequence[float]]] = None,
            weight_decay: float = 0.,
            **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        """
        Only take care about synaptic decay and possible range of synaptic
        weights.

        Returns
        -------
        None

        """
        super().update()


class STDP(LearningRule):
    """
    Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
            self,
            connection: AbstractConnection,
            lr: Optional[Union[float, Sequence[float]]] = None,
            weight_decay: float = 0.,
            **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        Consider the additional required parameters and fill the body\
        accordingly.
        """

        if isinstance(self.connection, (RandomConnection, DenseConnection)):
            self.update = self._update
        elif isinstance(self.connection, (ConvolutionalConnection, TorchConvolutionalConnection)):
            self.update = self.conv_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _update(self, **kwargs) -> None:
        """

        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """
        self.connection.w -= self.connection.pre.s.unsqueeze(1).float() @ (
                self.connection.post.traces.unsqueeze(0) * self.lr[0])
        self.connection.w += self.connection.pre.traces.unsqueeze(1) @ (
                self.connection.post.s.unsqueeze(0).float() * self.lr[1])
        super(STDP, self).update()

    def conv_update(self, **kwargs) -> None:
        """

        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """
        winners = kwargs.get('winners', None)
        assert winners is not None, 'Winners list required.'

        for winner in winners:
            post_s = self.connection.post.s[winner[0], winner[1], winner[2]].flatten()
            post_traces = self.connection.post.traces[winner[0], winner[1], winner[2]].flatten()
            pre_s = self.connection.pre.s[0, (winner[1]*self.connection.stride):(winner[1]*self.connection.stride) + self.connection.kernel_size,
                     (winner[2]*self.connection.stride):(winner[2]*self.connection.stride) + self.connection.kernel_size].flatten()
            pre_traces = self.connection.pre.traces[0, (winner[1]*self.connection.stride):(winner[1]*self.connection.stride) + self.connection.kernel_size,
                          (winner[2]*self.connection.stride):(winner[2]*self.connection.stride) + self.connection.kernel_size].flatten()
            # print(post_s.shape, post_traces.shape, pre_s.shape, pre_traces.shape)
            self.connection.w[winner[0]] -= (pre_s.unsqueeze(1).float() @ (
                    post_traces.unsqueeze(0) * self.lr[0])).reshape(self.connection.w[winner[0]].shape)
            self.connection.w[winner[0]] += (pre_traces.unsqueeze(1) @ (
                    post_s.unsqueeze(0).float() * self.lr[1])).reshape(self.connection.w[winner[0]].shape)
            super(STDP, self).update()


class FlatSTDP(LearningRule):
    """
    Flattened Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of Flat-STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
            self,
            connection: AbstractConnection,
            lr: Optional[Union[float, Sequence[float]]] = None,
            weight_decay: float = 0.,
            **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        Consider the additional required parameters and fill the body\
        accordingly.
        """
        self.window = kwargs.get('window', 10)
        if isinstance(self.connection, (RandomConnection, DenseConnection)):
            self.update = self._update
        elif isinstance(self.connection, (ConvolutionalConnection, TorchConvolutionalConnection)):
            self.update = self.conv_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )
    def _update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """

        self.connection.w -= self.connection.pre.s.unsqueeze(1).float() @ (
                torch.where(self.connection.post.traces.unsqueeze(0) > torch.pow(self.connection.post.trace_decay,
                                                                                 self.window / self.connection.post.dt),
                            1.0, 0.0) * self.lr[0])
        self.connection.w += torch.where(
            self.connection.pre.traces.unsqueeze(1) > torch.pow(self.connection.pre.trace_decay,
                                                                self.window / self.connection.pre.dt), 1.0, 0.0) @ (
                                     self.connection.post.s.unsqueeze(0).float() * self.lr[1])
        super(FlatSTDP, self).update()

    def conv_update(self, **kwargs) -> None:
        """

        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """
        winners = kwargs.get('winners', None)
        assert winners is not None, 'Winners list required.'

        for winner in winners:
            post_s = self.connection.post.s[winner[0], winner[1], winner[2]].flatten()
            post_traces = self.connection.post.traces[winner[0], winner[1], winner[2]].flatten()
            pre_s = self.connection.pre.s[0, (winner[1]*self.connection.stride):(winner[1]*self.connection.stride) + self.connection.kernel_size,
                     (winner[2]*self.connection.stride):(winner[2]*self.connection.stride) + self.connection.kernel_size].flatten()
            pre_traces = self.connection.pre.traces[0, (winner[1]*self.connection.stride):(winner[1]*self.connection.stride) + self.connection.kernel_size,
                          (winner[2]*self.connection.stride):(winner[2]*self.connection.stride) + self.connection.kernel_size].flatten()
            # print(post_s.shape, post_traces.shape, pre_s.shape, pre_traces.shape)

            self.connection.w[winner[0]] -= (pre_s.unsqueeze(1).float() @ (
                    torch.where(post_traces.unsqueeze(0) > torch.pow(self.connection.post.trace_decay,
                                                                                     self.window / self.connection.post.dt),
                                1.0, 0.0) * self.lr[0])).reshape(self.connection.w[winner[0]].shape)
            self.connection.w[winner[0]] += (torch.where(
                pre_traces.unsqueeze(1) > torch.pow(self.connection.pre.trace_decay,
                                                                    self.window / self.connection.pre.dt), 1.0, 0.0) @ (
                                         post_s.unsqueeze(0).float() * self.lr[1])).reshape(self.connection.w[winner[0]].shape)
            super(FlatSTDP, self).update()


class RSTDP(LearningRule):
    """
    Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
            self,
            connection: AbstractConnection,
            lr: Optional[Union[float, Sequence[float]]] = None,
            weight_decay: float = 0.,
            tau_c: int = 5,
            **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        Consider the additional required parameters and fill the body\
        accordingly.
        """
        self.c = torch.zeros(*self.connection.w.shape)
        self.tau_c = tau_c

    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        dt = self.connection.pre.dt
        dopamine = kwargs["dopamine"]
        self.c += (-self.c / self.tau_c) * dt
        self.c -= self.connection.pre.s.unsqueeze(1).float() @ (
                self.connection.post.traces.unsqueeze(0) * self.lr[0]) * dt
        self.c += self.connection.pre.traces.unsqueeze(1) @ (
                self.connection.post.s.unsqueeze(0).float() * self.lr[1]) * dt
        self.connection.w += dopamine * self.c * dt

        super(RSTDP, self).update()


class FlatRSTDP(LearningRule):
    """
    Flattened Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of Flat-RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
            self,
            connection: AbstractConnection,
            lr: Optional[Union[float, Sequence[float]]] = None,
            weight_decay: float = 0.,
            **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        Consider the additional required parameters and fill the body\
        accordingly.
        """
        self.window = kwargs.get('window', 10)
        self.c = torch.zeros(*self.connection.w.shape)
    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """


        dopamine = kwargs["dopamine"]

        self.connection.w -= self.connection.pre.s.unsqueeze(1).float() @ (
                torch.where(self.connection.post.traces.unsqueeze(0) > torch.pow(self.connection.post.trace_decay,
                                                                                 self.window / self.connection.post.dt),
                            1.0, 0.0) * self.lr[0]) * sign(dopamine)
        self.connection.w += torch.where(
            self.connection.pre.traces.unsqueeze(1) > torch.pow(self.connection.pre.trace_decay,
                                                                self.window / self.connection.pre.dt), 1.0, 0.0) @ (
                                     self.connection.post.s.unsqueeze(0).float() * self.lr[1]) * sign(dopamine)

        super(FlatRSTDP, self).update()
