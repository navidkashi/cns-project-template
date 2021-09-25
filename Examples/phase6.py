from cnsproject.network.neural_populations import LIFPopulation, InputPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection, RandomConnection
from cnsproject.network.network import Network
from cnsproject.encoding.encoders import *
from cnsproject.learning.learning_rules import *
from cnsproject.utils import *
import matplotlib.pyplot as plt
import torch


from cnsproject.plotting.plotting import plot_potential, plot_traces, plot_weights, plot_spikes, plot_params, \
    plot_default_params

time = 2000  # time of simulation in ms
dt = 1.0  # dt in ms

input_layer = InputPopulation(shape=(4,), dt=dt, additive_spike_trace=True, tau_s=100)
first_layer = LIFPopulation(shape=(2,), v_th=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m=50, r_m=10, t_ref=0,
                       init_t_ref=0, dt=dt, additive_spike_trace=True, tau_s=100)

monitor_input_layer = Monitor(input_layer, state_variables=["s", "traces"])
monitor_first_layer = Monitor(first_layer, state_variables=["s", "traces", "v"])


connection = DenseConnection(input_layer, first_layer, w=torch.rand(input_layer.s.numel(), first_layer.s.numel()) * 0.3, learning_rule=STDP, lr=[0.1, 0.1], weight_decay=0, wmax=4)
monitor_w = Monitor(connection, state_variables=["w"])

net = Network(learning=True, dt=dt)
net.add_layer(input_layer, "input")
net.add_layer(first_layer, "first")

net.add_connection(connection, "input", "first")

net.add_monitor(monitor_input_layer, "input")
net.add_monitor(monitor_first_layer, "first")
net.add_monitor(monitor_w, "connection")

enc = PoissonEncoder(time / 4, dt, max_rate=100)
# img1 = [1, 0.9, .9, .9, 1, 0.01, .01, .01, .01, .01]
# img2 = [0.01, .01, .01, .01, .01, 1, 0.9, .9, .9, 1]
img1 = [.01, .01, .01, 1]
img2 = [1, .01, .01, .01]
encoded2 = enc(torch.tensor(img1, dtype=torch.float)).float()
encoded1 = enc(torch.tensor(img2, dtype=torch.float)).float()
# perm = torch.randperm(random_period * 2)

# encoded_random = torch.cat((encoded1, encoded2))[perm[:random_period]]
# encoded = torch.cat((encoded_random, encoded1[:int((time - random_period) / 2)], encoded2[:int((time - random_period) / 2)]))
encoded = torch.cat((encoded1, encoded2, encoded1, encoded2))
# print(torch.bernoulli(torch.ones(150) * 0.5).long())
inputs = {'input': encoded}

net.run(time, inputs)

s_input = monitor_input_layer.get("s")
traces_input = monitor_input_layer.get("traces")
s_first = monitor_first_layer.get("s")
traces_first = monitor_first_layer.get("traces")
v = monitor_first_layer.get("v")
w = monitor_w.get("w")
plt.rcParams["figure.figsize"] = 14, 20
fig = plt.figure()
axs = fig.subplots(6, gridspec_kw={'hspace': 0.5})
plot_spikes(axs[0], 'Input Spikes', s_input, dt)
plot_spikes(axs[1], 'Output Spikes', s_first, dt)
plot_traces(axs[2], 'Input Traces', traces_input, dt)
plot_traces(axs[3], 'Output Traces', traces_first, dt)
# plot_potential(axs[4], 'Output Voltage', v, dt)
plot_weights(axs[4], 'Post neuron 0 weights', w[:, :, 0].reshape(time, -1), dt)
plot_weights(axs[5], 'Post neuron 1 weights', w[:, :, 1].reshape(time, -1), dt)
legend0 = ['Pre neuron {:d}'.format(i) for i in range(w[0, :, 0].numel())]
legend1 = ['Pre neuron {:d}'.format(i) for i in range(w[0, :, 1].numel())]
axs[4]. legend(legend0, loc='upper left', frameon=False)
axs[5]. legend(legend1, loc='upper left', frameon=False)
fig.show()
fig.savefig('stdp_add_single_3_1_1_tau.png', bbox_inches='tight')

# plt.rcParams["figure.figsize"] = 14, 20
# fig = plt.figure()
# axs = fig.subplots(2, gridspec_kw={'hspace': 0.5})
# plot_weights(axs[0], 'Post neuron 1 weight differences', w[:, :, 1].reshape(time, -1)[1:] - w[:, :, 1].reshape(time, -1)[:-1], dt)
# fig.show()
# low rate
#2 both high rate
#3 inh high rate
#4 exc high rate