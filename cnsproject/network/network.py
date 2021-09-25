"""
Module for spiking neural network construction and simulation.
"""

from typing import Optional, Dict

import torch

from .neural_populations import NeuralPopulation
from .connections import AbstractConnection
from .monitors import Monitor
from ..learning.rewards import AbstractReward
from ..decision.decision import AbstractDecision


class Network(torch.nn.Module):
    """
    The class responsible for creating a neural network and its simulation.

    Examples
    --------
    >>> from network.neural_populations import LIFPopulation
    >>> from network.connections import DenseConnection
    >>> from network.monitors import Monitor
    >>> from network import Network
    >>> inp = InputPopulation(shape=(10,))
    >>> out = LIFPopulation(shape=(2,))
    >>> synapse = DenseConnection(inp, out)
    >>> net = Network(learning=False)
    >>> net.add_layer(inp, "input")
    >>> net.add_layer(out, "output")
    >>> net.add_connection(synapse, "input", "output")
    >>> out_m = Monitor(out, state_variables=["s", "v"])
    >>> syn_m = Monitor(synapse, state_variables=["w"])  # `w` indicates synaptic weights
    >>> net.add_monitor(out_m, "output")
    >>> net.add_monitor(syn_m, "synapse")
    >>> net.run(10)
    Here, we create a simple network with two layers and dense connection. We aim to monitor
    the synaptic weights and output layer's spikes and voltages. We simulate the network for
    10 miliseconds.

    You will need to implement the `run` method. This mthod is responsible for the whole simulation \
    procedure of a spiking neural network. You will have to compute number of time steps using \
    `dt` attribute of the class and `time` parameter of the method. then you will iteratively call \
    the procedures for single step simulation of network objects.

    **NOTE:** If you faced any errors related to importing packages, modify the `__init__.py` files \
    accordingly to solve the problem.

    Arguments
    ---------
    dt : float, Optional
        Specify simulation timestep. The default is 1.0.
    learning: bool, Optional
        Whether to allow weight update and learning. The default is True.
    reward : AbstractReward, Optional
        The class to allow reward modifications in case of reward-modulated
        learning. The default is None.
    decision: AbstractDecision, Optional
        The class to enable decision making. The default is None.

    """

    def __init__(
            self,
            dt: float = 1.0,
            learning: bool = True,
            reward: Optional[AbstractReward] = None,
            decision: Optional[AbstractDecision] = None,
            **kwargs
    ) -> None:
        super().__init__()

        self.dt = dt

        self.layers = {}
        self.connections = {}
        self.monitors = {}

        self.train(learning)

        # Make sure that arguments of your reward and decision classes do not
        # share same names. Their arguments are passed to the network as its
        # keyword arguments.
        if reward is not None:
            self.reward = reward(**kwargs)
        else:
            self.reward = None

        if decision is not None:
            self.decision = decision(**kwargs)
        else:
            self.decision = None

        self.dopamine = None
        self.actual_reward = None
        self.predicted_reward = None


    def add_layer(self, layer: NeuralPopulation, name: str) -> None:
        """
        Add a neural population to the network.

        Parameters
        ----------
        layer : NeuralPopulation
            The neural population to be added.
        name : str
            Name of the layer for further referencing.

        Returns
        -------
        None

        """
        self.layers[name] = layer
        self.add_module(name, layer)
        layer.train(self.learning)
        layer.dt = self.dt

    def add_connection(
            self,
            connection: AbstractConnection,
            pre: str,
            post: str
    ) -> None:
        """
        Add a connection between neural populations to the network. The\
        reference name will be in the format `{pre}_to_{post}`.

        Parameters
        ----------
        connection : AbstractConnection
            The connection to be added.
        pre : str
            Reference name of pre-synaptic population.
        post : str
            Reference name of post-synaptic population.

        Returns
        -------
        None

        """
        self.connections[f"{pre}_to_{post}"] = connection
        self.add_module(f"{pre}_to_{post}", connection)

        connection.train(self.learning)
        connection.dt = self.dt

    def add_monitor(self, monitor: Monitor, name: str) -> None:
        """
        Add a monitor on a network object to the network.

        Parameters
        ----------
        monitor : Monitor
            The monitor instance to be added.
        name : str
            Name of the monitor instance for further referencing.

        Returns
        -------
        None

        """
        self.monitors[name] = monitor
        # monitor.dt = self.dt

    def run(
            self,
            time: int,
            inputs: Dict[str, torch.Tensor] = None,
            one_step: bool = False,
            **kwargs
    ) -> None:
        """
        Simulate network for a specific time duration with the possible given\
        input.

        Input to each layer is given to `inputs` parameter. As you see, it is a \
        dictionary of population's name and tensor of input values through time. \
        There is a parameter named `one_step`. This parameter will define how the \
        input is propagated through the network: does it go forward up to the final \
        layer in one time step or it passes from one layer to the next in each \
        step of simulation. You can easily remove it if it is mind-bugling.

        Also, make sure to call `self.reset_state_variables()` before starting the \
        simulation.

        TODO.

        Implement the body of this method.

        Parameters
        ----------
        time : int
            Simulation time.
        inputs : Dict[str, torch.Tensor], optional
            Mapping of input layer names to their input spike tensors. The\
            default is {}.
        one_step : bool, optional
            Whether to propagate the inputs all the way through the network in\
            a single simulation step. The default is False.

        Keyword Arguments
        -----------------
        clamp : Dict[str, torch.Tensor]
            Mapping of layer names to boolean masks if neurons should be clamped
            to spiking.
        unclamp : Dict[str, torch.Tensor]
            Mapping of layer names to boolean masks if neurons should be clamped
            not to spiking.
        masks : Dict[str, torch.Tensor]
            Mapping of connection names to boolean masks of the weights to clamp
            to zero.

        **Note:** you can pass the reward and decision methods' arguments as keyword\
        arguments to this function.

        Returns
        -------
        None

        """
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        env_reward = kwargs.get("env_reward", None)
        kwargs["dopamine"] = 0
        self.reset_state_variables()

        for t in range(int(time / self.dt)):

            for layer in inputs:
                if layer in self.layers:
                    self.layers[layer].forward(inputs[layer][t])

            potential_inputs = {}
            for con in self.connections:
                # Check whether name of post population in connection definition is in layers or not.
                post = con.split('_to_')[1]
                if post in self.layers:
                    # Update potential injection on post population.
                    if post in potential_inputs:
                        potential_inputs[post] += self.connections[con].compute()
                    else:
                        potential_inputs[post] = self.connections[con].compute()

            if self.decision is not None:
                decision_pop_name = kwargs.get('decision_pop', None)
                assert decision_pop_name is not None, 'Decision Population required.'
                decision_pop = self.layers.get(decision_pop_name, None)
                assert decision_pop is not None, 'Decision Population not found.'
                done, winners, inhibition_pot = self.decision.compute(decision_pop, **kwargs)
                kwargs['winners'] = winners

                if decision_pop_name in potential_inputs:
                    potential_inputs[decision_pop_name] += inhibition_pot
                else:
                    potential_inputs[decision_pop_name] = inhibition_pot

            for con in self.connections:
                self.connections[con].update(**kwargs)


            for layer in potential_inputs:
                self.layers[layer].inject_v = potential_inputs[layer]
            for layer in self.layers:
                if layer not in inputs:
                    self.layers[layer].forward(torch.zeros(*self.layers[layer].s.shape))

            if env_reward is not None:
                kwargs["actual_reward"] = env_reward(self.layers["input"].s, self.layers["output"].s)
                kwargs["dopamine"] += (self.reward.compute(**kwargs) - (kwargs["dopamine"] / kwargs.get("tau_d", 10))) * self.dt

                self.dopamine = torch.tensor(kwargs["dopamine"]) # for plotting only
            # print(t)




            for m in self.monitors:
                self.monitors[m].record()

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

        for monitor in self.monitors:
            self.monitors[monitor].reset_state_variables()

        if self.decision is not None:
            self.decision.reset_state_variables()

    def train(self, mode: bool = True) -> "torch.nn.Moudle":
        """
        Set the population's training mode.

        Parameters
        ----------
        mode : bool, optional
            Mode of training. `True` turns on the training while `False` turns\
            it off. The default is True.

        Returns
        -------
        torch.nn.Module

        """
        self.learning = mode
        return super().train(mode)
