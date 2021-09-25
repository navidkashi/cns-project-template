"""
Module for connections between neural populations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Tuple

import torch

from .neural_populations import NeuralPopulation
import torch.nn.functional as F

class AbstractConnection(ABC, torch.nn.Module):
    """
    Abstract class for implementing connections.

    Make sure to implement the `compute`, `update`, and `reset_state_variables`\
    methods in your child class.

    You will need to define the populations you want to connect as `pre` and `post`.\
    In case of learning, you will need to define the learning rate (`lr`) and the \
    learning rule to follow. Attribute `w` is reserved for synaptic weights.\
    However, it has not been predefined or allocated, as it depends on the \
    pattern of connectivity. So make sure to define it in child class initializations \
    appropriately to indicate the pattern of connectivity. The default range of \
    each synaptic weight is [0, 1] but it can be controlled by `wmin` and `wmax`. \
    Synaptic strengths might decay in time and do not last forever. To define \
    the decay rate of the synaptic weights, use `weight_decay` attribute. Also, \
    if you want to control the overall input synaptic strength to each neuron, \
    use `norm` argument to normalize the synaptic weights.

    In case of learning, you have to implement the methods `compute` and `update`. \
    You will use the `compute` method to calculate the activity of post-synaptic \
    population based on the pre-synaptic one. Update of weights based on the \
    learning rule will be implemented in the `update` method. If you find this \
    architecture mind-bugling, try your own architecture and make sure to redefine \
    the learning rule architecture to be compatible with this new architecture \
    of yours.

    Arguments
    ---------
    pre : NeuralPopulation
        The pre-synaptic neural population.
    post : NeuralPopulation
        The post-synaptic neural population.
    lr : float or (float, float), Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float, Optional
        Define rate of decay in synaptic strength. The default is 0.0.

    Keyword Arguments
    -----------------
    learning_rule : LearningRule
        Define the learning rule by which the network will be trained. The\
        default is NoOp (see learning/learning_rules.py for more details).
    wmin : float
        The minimum possible synaptic strength. The default is 0.0.
    wmax : float
        The maximum possible synaptic strength. The default is 1.0.
    norm : float
        Define a normalization on input signals to a population. If `None`,\
        there is no normalization. The default is None.

    """

    def __init__(
            self,
            pre: NeuralPopulation,
            post: NeuralPopulation,
            lr: Union[float, Sequence[float]] = None,
            weight_decay: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__()

        assert isinstance(pre, NeuralPopulation), \
            "Pre is not a NeuralPopulation instance"
        assert isinstance(post, NeuralPopulation), \
            "Post is not a NeuralPopulation instance"

        self.pre = pre
        self.post = post

        self.weight_decay = weight_decay

        self.wmin = kwargs.get('wmin', 0.)
        self.wmax = kwargs.get('wmax', 4)
        self.norm = kwargs.get('norm', None)


        from ..learning.learning_rules import NoOp

        learning_rule = kwargs.get('learning_rule', NoOp)

        self.learning_rule = learning_rule(
            connection=self,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

    @abstractmethod
    def compute(self, s: torch.Tensor) -> None:
        """
        Compute the post-synaptic neural population activity based on the given\
        spikes of the pre-synaptic population.

        Parameters
        ----------
        s : torch.Tensor
            The pre-synaptic spikes tensor.

        Returns
        -------
        None

        """

        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Compute connection's learning rule and weight update.

        Keyword Arguments
        -----------------
        learning : bool
            Whether learning is enabled or not. The default is True.
        mask : torch.ByteTensor
            Define a mask to determine which weights to clamp to zero.

        Returns
        -------
        None

        """
        learning = kwargs.get("learning", True)

        if learning:
            self.learning_rule.update(**kwargs)

        mask = kwargs.get("mask", None)
        if mask is not None:
            self.w.masked_fill_(mask, 0)

    @abstractmethod
    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        pass


class DenseConnection(AbstractConnection):
    """
    Specify a fully-connected synapse between neural populations.

    Implement the dense connection pattern following the abstract connection\
    template.
    """

    def __init__(
            self,
            pre: NeuralPopulation,
            post: NeuralPopulation,
            lr: Union[float, Sequence[float]] = None,
            weight_decay: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.wmin + torch.rand(pre.s.numel(), post.s.numel()) * (self.wmax - self.wmin)
        self.w = kwargs.get("w", self.wmin + torch.rand(pre.s.numel(), post.s.numel()) * (self.wmax - self.wmin))

    def compute(self, s: torch.Tensor = None) -> torch.Tensor:
        """
        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """
        if self.pre.is_inhibitory:
            post_input = torch.matmul(self.pre.s.float(), -self.w / (self.post.s.numel() / 100.)).view(
                *self.post.s.shape)
        else:
            post_input = torch.matmul(self.pre.s.float(), self.w / (self.post.s.numel() / 100.)).view(
                *self.post.s.shape)
        return post_input

    def update(self, **kwargs) -> None:
        """

        Update the connection weights based on the learning rule computations.\
        You might need to call the parent method.
        """
        super(DenseConnection, self).update(**kwargs)

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        super(DenseConnection, self).reset_state_variables()


class RandomConnection(AbstractConnection):
    """
    Specify a random synaptic connection between neural populations.

    Implement the random connection pattern following the abstract connection\
    template.
    """

    def __init__(
            self,
            pre: NeuralPopulation,
            post: NeuralPopulation,
            connection_rate: float,
            lr: Union[float, Sequence[float]] = None,
            weight_decay: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        super().__init__(pre, post, lr, weight_decay, **kwargs)
        assert 0. < connection_rate < 1., \
            "Connection rate must be in (0, 1)"

        self.w = kwargs.get("w", None)
        if self.w is None:
            self.w = self.wmin + torch.rand(pre.s.numel(), post.s.numel()) * (self.wmax - self.wmin)

        n = pre.s.numel() * post.s.numel()
        connected_idx = torch.randperm(n)[:round(n * connection_rate)]
        self.rand_mask = torch.flatten(torch.ones(pre.s.numel(), post.s.numel())).scatter_(0, connected_idx, 0).view(
            pre.s.numel(), post.s.numel()).bool()
        self.w.masked_fill_(self.rand_mask, 0)

    def compute(self, s: torch.Tensor = None) -> torch.Tensor:
        """

        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """
        if self.pre.is_inhibitory:
            post_input = torch.matmul(self.pre.s.float(), -self.w / (self.post.s.numel() / 100.)).view(
                *self.post.s.size)
        else:
            post_input = torch.matmul(self.pre.s.float(), self.w / (self.post.s.numel() / 100.)).view(*self.post.s.size)
        return post_input

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.\
        You might need to call the parent method.
        """
        super(RandomConnection, self).update(**kwargs)

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class ConvolutionalConnection(AbstractConnection):
    """
    Specify a convolutional synaptic connection between neural populations.

    Implement the convolutional connection pattern following the abstract\
    connection template.
    """

    def __init__(
            self,
            pre: NeuralPopulation,
            post: NeuralPopulation,
            kernel_size: int,
            padding: bool = True,
            stride: int = 1,
            lr: Union[float, Sequence[float]] = None,
            weight_decay: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.c_in = pre.shape[0]
        self.h_in = pre.shape[1]
        self.w_in = pre.shape[2]
        self.c_out = post.shape[0]
        self.h_out = post.shape[1]
        self.w_out = post.shape[2]
        if self.padding:
            self.pad_size = self.kernel_size // 2
        else:
            self.pad_size = 0

        self.output_shape = (self.c_out, (self.h_in - kernel_size + self.pad_size * 2) // self.stride + 1,
                        (self.w_in - kernel_size + self.pad_size * 2) // self.stride + 1)

        assert (post.shape[1] == self.output_shape[1] and post.shape[2] == self.output_shape[2]), "Wrong post population shape. Must be " + str(self.output_shape)

        self.w = kwargs.get("w", self.wmin + torch.rand(self.c_out, self.c_in, self.kernel_size, self.kernel_size) * (self.wmax - self.wmin))


    def compute(self, s: torch.Tensor = None) -> torch.Tensor:
        """
        TODO.

        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """
        kernel = self.w.clone()
        image = self.pre.s.clone()
        if self.padding:
            padded_image = torch.zeros((image.shape[0], image.shape[1] + self.pad_size * 2, image.shape[2] + self.pad_size * 2))
            padded_image[:, self.pad_size:-self.pad_size, self.pad_size:-self.pad_size] = image
        else:
            padded_image = image.clone()
        kernel_means = torch.mean(kernel, [2, 3])
        output = torch.zeros(self.output_shape)
        for c_out in range(self.c_out):
            for c_in in range(self.c_in):
                for i in range(self.output_shape[1]):
                    for j in range(self.output_shape[2]):
                        output[c_out, i, j] += torch.sum(
                            torch.multiply(padded_image[c_in, i*self.stride:i*self.stride + kernel.shape[2], j*self.stride:j*self.stride + kernel.shape[3]], kernel[c_out, c_in] - kernel_means[c_out, c_in]))
        return output

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.
        You might need to call the parent method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class PoolingConnection(AbstractConnection):
    """
    Specify a pooling synaptic connection between neural populations.

    Implement the pooling connection pattern following the abstract connection\
    template. Consider a parameter for defining the type of pooling.

    Note: The pooling operation does not support learning. You might need to\
    make some modifications in the defined structure of this class.
    """

    def __init__(
            self,
            pre: NeuralPopulation,
            post: NeuralPopulation,
            kernel_size: int,
            padding: bool = True,
            stride: int = 1,
            trace_scale: float = 1.0,
            lr: Union[float, Sequence[float]] = None,
            weight_decay: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.kernel_size = kernel_size
        self.decay = kwargs.get('decay', 0.0)
        self.firing_trace = torch.zeros(self.pre.s.shape)
        self.padding = padding
        self.stride = stride
        self.trace_scale = trace_scale
        self.c_in = pre.shape[0]
        self.h_in = pre.shape[1]
        self.w_in = pre.shape[2]
        self.c_out = post.shape[0]
        self.h_out = post.shape[1]
        self.w_out = post.shape[2]
        if self.padding:
            self.pad_size = self.kernel_size // 2
        else:
            self.pad_size = 0

        self.output_shape = (self.c_out, (self.h_in - kernel_size + self.pad_size * 2) // self.stride + 1,
                        (self.w_in - kernel_size + self.pad_size * 2) // self.stride + 1)

        assert (self.c_out == self.c_in and post.shape[1] == self.output_shape[1] and post.shape[2] == self.output_shape[2]), "Wrong post population shape. Must be " + str(self.output_shape)


    def compute(self, s: torch.Tensor = None) -> torch.Tensor:
        """
        TODO.

        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """
        self.firing_trace *= (1. - self.decay)
        self.firing_trace += self.pre.s * self.trace_scale
        image = self.firing_trace.clone()
        if self.padding:
            padded_image = torch.zeros(
                (image.shape[0], image.shape[1] + self.pad_size * 2, image.shape[2] + self.pad_size * 2))
            padded_image[:, self.pad_size:-self.pad_size, self.pad_size:-self.pad_size] = image
        else:
            padded_image = image.clone()
        output = torch.zeros(self.output_shape)
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                output[:, i, j] = torch.max(torch.max(padded_image[:, i*self.stride:i*self.stride + self.kernel_size, j*self.stride:j*self.stride + self.kernel_size], 1)[0], 1)[0][:]

        return output

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.\
        You might need to call the parent method.

        Note: You should be careful with this method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass

class TorchConvolutionalConnection(AbstractConnection):
    """
    Specify a convolutional synaptic connection between neural populations.

    Implement the convolutional connection pattern following the abstract\
    connection template.
    """

    def __init__(
            self,
            pre: NeuralPopulation,
            post: NeuralPopulation,
            kernel_size: int,
            padding: bool = True,
            stride: int = 1,
            lr: Union[float, Sequence[float]] = None,
            weight_decay: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.c_in = pre.shape[0]
        self.h_in = pre.shape[1]
        self.w_in = pre.shape[2]
        self.c_out = post.shape[0]
        self.h_out = post.shape[1]
        self.w_out = post.shape[2]
        if self.padding:
            self.pad_size = self.kernel_size // 2
        else:
            self.pad_size = 0

        self.output_shape = (self.c_out, (self.h_in - kernel_size + self.pad_size * 2) // self.stride + 1,
                        (self.w_in - kernel_size + self.pad_size * 2) // self.stride + 1)

        assert (post.shape[1] == self.output_shape[1] and post.shape[2] == self.output_shape[2]), "Wrong post population shape. Must be " + str(self.output_shape)

        self.w = kwargs.get("w", self.wmin + torch.rand(self.c_out, self.c_in, self.kernel_size, self.kernel_size) * (self.wmax - self.wmin))


    def compute(self, s: torch.Tensor = None) -> torch.Tensor:
        """
        TODO.

        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """
        output = F.conv2d(
            self.pre.s.float().unsqueeze(0),
            self.w,
            stride=(self.stride, self.stride),
            padding=(self.pad_size, self.pad_size),
        ).squeeze(0)
        # print(output.shape)
        return output

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.
        You might need to call the parent method.
        """
        super(TorchConvolutionalConnection, self).update(**kwargs)

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class TorchPoolingConnection(AbstractConnection):
    """
    Specify a pooling synaptic connection between neural populations.

    Implement the pooling connection pattern following the abstract connection\
    template. Consider a parameter for defining the type of pooling.

    Note: The pooling operation does not support learning. You might need to\
    make some modifications in the defined structure of this class.
    """

    def __init__(
            self,
            pre: NeuralPopulation,
            post: NeuralPopulation,
            kernel_size: int,
            padding: bool = True,
            stride: int = 1,
            trace_scale: float = 1.0,
            lr: Union[float, Sequence[float]] = None,
            weight_decay: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.kernel_size = kernel_size
        self.decay = kwargs.get('decay', 0.0)
        self.firing_trace = torch.zeros(self.pre.s.shape)
        self.padding = padding
        self.stride = stride
        self.trace_scale = trace_scale
        self.c_in = pre.shape[0]
        self.h_in = pre.shape[1]
        self.w_in = pre.shape[2]
        self.c_out = post.shape[0]
        self.h_out = post.shape[1]
        self.w_out = post.shape[2]
        if self.padding:
            self.pad_size = self.kernel_size // 2
        else:
            self.pad_size = 0

        self.output_shape = (self.c_out, (self.h_in - kernel_size + self.pad_size * 2) // self.stride + 1,
                        (self.w_in - kernel_size + self.pad_size * 2) // self.stride + 1)

        assert (self.c_out == self.c_in and post.shape[1] == self.output_shape[1] and post.shape[2] == self.output_shape[2]), "Wrong post population shape. Must be " + str(self.output_shape)


    def compute(self, s: torch.Tensor = None) -> torch.Tensor:
        """
        TODO.

        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """
        self.firing_trace *= (1. - self.decay)
        self.firing_trace += self.pre.s.float() * self.trace_scale
        # print(self.firing_trace.unsqueeze(1).shape)
        output, indices = F.max_pool2d(
            self.firing_trace.unsqueeze(1),
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(self.stride, self.stride),
            padding=(self.pad_size, self.pad_size),
            return_indices=True
        )
        # print(output)
        # output = torch.zeros(self.output_shape)
        # output[0, indices.squeeze(0)[0]] = 10.
        # print(output.shape)
        return output.squeeze(0)
        # s = self.pre.s.clone()
        # return s.flatten(2).gather(2, indices.flatten(2)).view_as(indices).float()

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.\
        You might need to call the parent method.

        Note: You should be careful with this method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass

