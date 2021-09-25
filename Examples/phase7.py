from cnsproject.network.neural_populations import LIFPopulation, InputPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection, RandomConnection
from cnsproject.network.network import Network
from cnsproject.encoding.encoders import *
from cnsproject.learning.learning_rules import *
from cnsproject.learning.rewards import RawReward, RPE

from cnsproject.utils import *
import matplotlib.pyplot as plt
import torch


from cnsproject.plotting.plotting import plot_value, plot_traces, plot_weights, plot_spikes, plot_params, \
    plot_default_params

def env_reward(input, output):
    if output[0] and not output[1]:
        if torch.sum(input * torch.tensor([0, 1, 0, 1])) >= 1:
            return 1.0
        else:
            return -1.0
    elif output[1] and not output[0]:
        if torch.sum(input * torch.tensor([1, 0, 1, 0])) >= 1:
            return 1.0
        else:
            return -1.0
    elif output[1] and output[0]:
        return -1.0
    else:
        return 0.0

time = 5000  # time of simulation in ms
dt = 1.0  # dt in ms
episods = 10
delay = 100
stimuli_time = time - (episods-1) * delay

input_layer = InputPopulation(shape=(4,), dt=dt, additive_spike_trace=True, tau_s=10)
first_layer = LIFPopulation(shape=(2,), v_th=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m=50, r_m=10, t_ref=0,
                       init_t_ref=0, dt=dt, additive_spike_trace=True, tau_s=10)

monitor_input_layer = Monitor(input_layer, state_variables=["s", "traces"])
monitor_first_layer = Monitor(first_layer, state_variables=["s", "traces", "v"])


connection = DenseConnection(input_layer, first_layer, w=torch.rand(input_layer.s.numel(), first_layer.s.numel()) * 0.5,
                             learning_rule=RSTDP, lr=[0.05, 0.05], tau_c=10, weight_decay=0, wmax=4)
connection_monitor = Monitor(connection, state_variables=["w"])

net = Network(learning=True, dt=dt, reward=RPE)
net.add_layer(input_layer, "input")
net.add_layer(first_layer, "output")

net.add_connection(connection, "input", "output")

net.add_monitor(monitor_input_layer, "input")
net.add_monitor(monitor_first_layer, "output")
net.add_monitor(connection_monitor, "connection")

network_monitor = Monitor(net, state_variables=["dopamine"])
net.add_monitor(network_monitor, "network")

leaning_rule_monitor = Monitor(connection.learning_rule, state_variables=["c"])
net.add_monitor(leaning_rule_monitor, "leaning_rule")


enc = PoissonEncoder(stimuli_time / episods, dt, max_rate=30)

img1 = [0, 1, 0, 1]
img2 = [1, 0, 1, 0]
encoded2 = enc(torch.tensor(img1, dtype=torch.float)).float()
encoded1 = enc(torch.tensor(img2, dtype=torch.float)).float()
delay_encode = torch.zeros(delay, *encoded1[0].shape)

encoded = torch.cat((encoded1, delay_encode, encoded2))

for i in range(int(episods / 2 - 1)):
    encoded = torch.cat((encoded, delay_encode, encoded1, delay_encode, encoded2))

inputs = {'input': encoded}

net.run(time, inputs, env_reward=env_reward, tau_d=20)

s_input = monitor_input_layer.get("s")
traces_input = monitor_input_layer.get("traces")
s_first = monitor_first_layer.get("s")
traces_first = monitor_first_layer.get("traces")
v = monitor_first_layer.get("v")
w = connection_monitor.get("w")
dopamine = network_monitor.get("dopamine")
eligibility = leaning_rule_monitor.get("c")
plt.rcParams["figure.figsize"] = 16, 22
fig = plt.figure()
axs = fig.subplots(7, gridspec_kw={'hspace': 0.5})
plot_spikes(axs[0], 'Input Spikes', s_input, dt)
plot_spikes(axs[1], 'Output Spikes', s_first, dt)
plot_value(axs[2], 'Dopamine', dopamine, dt)
plot_value(axs[3], 'Post neuron 0 Eligibility', eligibility[:, :, 0].reshape(time, -1), dt)
plot_value(axs[4], 'Post neuron 1 Eligibility', eligibility[:, :, 1].reshape(time, -1), dt)
plot_weights(axs[5], 'Post neuron 0 weights', w[:, :, 0].reshape(time, -1), dt)
plot_weights(axs[6], 'Post neuron 1 weights', w[:, :, 1].reshape(time, -1), dt)
legend0 = ['Pre neuron {:d}'.format(i) for i in range(w[0, :, 0].numel())]
legend1 = ['Pre neuron {:d}'.format(i) for i in range(w[0, :, 1].numel())]
axs[3].legend(legend0, loc='upper left', frameon=False)
axs[4].legend(legend1, loc='upper left', frameon=False)
axs[5].legend(legend0, loc='upper left', frameon=False)
axs[6].legend(legend1, loc='upper left', frameon=False)
fig.show()
fig.savefig('rpe__10_50.png', bbox_inches='tight')

