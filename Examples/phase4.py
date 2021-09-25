from cnsproject.network.neural_populations import LIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection, RandomConnection
from cnsproject.network.network import Network
from cnsproject.utils import *
import matplotlib.pyplot as plt
import torch
import numpy as np

from cnsproject.plotting.plotting import plot_potential, plot_activity, plot_current, plot_spikes, plot_params, \
    plot_default_params

time = 100  # time of simulation in ms
dt = 0.03125  # dt in ms


n_exc1 = 500
n_exc2 = 500
n_inh = 500
n_pop = n_exc1 + n_exc2 + n_inh


exc1 = LIFPopulation(shape=(n_exc1,), v_th=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m=50, r_m=10, t_ref=0,
                       init_t_ref=0, dt=dt)
exc2 = LIFPopulation(shape=(n_exc2,), v_th=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m=50, r_m=10, t_ref=0,
                       init_t_ref=0, dt=dt)
inh = LIFPopulation(shape=(n_inh,), v_th=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m=50, r_m=10, t_ref=0,
                       init_t_ref=0, dt=dt, is_inhibitory=True)

monitor_exc1 = Monitor(exc1, state_variables=["s"])
monitor_exc2 = Monitor(exc2, state_variables=["s"])
monitor_inh = Monitor(inh, state_variables=["s"])


# exc1_exc1_connection = DenseConnection(exc1, exc1, w=torch.ones(exc1.s.numel(), exc1.s.numel()) * 0.05)
# exc2_exc2_connection = DenseConnection(exc2, exc2, w=torch.ones(exc2.s.numel(), exc2.s.numel()) * 0.05)
# exc1_inh_connection = DenseConnection(exc1, inh, w=torch.ones(exc1.s.numel(), inh.s.numel()) * 0.2)
# exc2_inh_connection = DenseConnection(exc2, inh, w=torch.ones(exc2.s.numel(), inh.s.numel()) * 0.2)
# inh_exc1_connection = DenseConnection(inh, exc1, w=torch.ones(inh.s.numel(), exc1.s.numel()) * 0.2)
# inh_exc2_connection = DenseConnection(inh, exc2, w=torch.ones(inh.s.numel(), exc2.s.numel()) * 0.2)

# exc1_exc1_connection = DenseConnection(exc1, exc1, w=torch.ones(exc1.s.numel(), exc1.s.numel()) * 0.05)
# exc2_exc2_connection = DenseConnection(exc2, exc2, w=torch.ones(exc2.s.numel(), exc2.s.numel()) * 0.05)
# exc1_inh_connection = RandomConnection(exc1, inh, connection_rate=0.6, w=torch.ones(exc1.s.numel(), inh.s.numel()) * 0.8)
# exc2_inh_connection = RandomConnection(exc2, inh, connection_rate=0.6, w=torch.ones(exc2.s.numel(), inh.s.numel()) * 0.8)
# inh_exc1_connection = DenseConnection(inh, exc1, w=torch.ones(inh.s.numel(), exc1.s.numel()) * 0.1)
# inh_exc2_connection = DenseConnection(inh, exc2, w=torch.ones(inh.s.numel(), exc2.s.numel()) * 0.1)

# exc1_exc1_connection = DenseConnection(exc1, exc1, w=torch.ones(exc1.s.numel(), exc1.s.numel()) * 0.05)
# exc2_exc2_connection = DenseConnection(exc2, exc2, w=torch.ones(exc2.s.numel(), exc2.s.numel()) * 0.05)
# exc1_inh_connection = RandomConnection(exc1, inh, connection_rate=0.6, w=torch.ones(exc1.s.numel(), inh.s.numel()) * 0.8)
# exc2_inh_connection = RandomConnection(exc2, inh, connection_rate=0.6, w=torch.ones(exc2.s.numel(), inh.s.numel()) * 0.8)
# inh_exc1_connection = RandomConnection(inh, exc1, connection_rate=0.6, w=torch.ones(inh.s.numel(), exc1.s.numel()) * 1)
# inh_exc2_connection = RandomConnection(inh, exc2, connection_rate=0.6, w=torch.ones(inh.s.numel(), exc2.s.numel()) * 1)

exc1_exc1_connection = RandomConnection(exc1, exc1, connection_rate=0.3, w=torch.ones(exc1.s.numel(), exc1.s.numel()) * 0.1)
exc2_exc2_connection = RandomConnection(exc2, exc2, connection_rate=0.3, w=torch.ones(exc2.s.numel(), exc2.s.numel()) * 0.1)
exc1_inh_connection = RandomConnection(exc1, inh, connection_rate=0.1, w=torch.ones(exc1.s.numel(), inh.s.numel()) * 0.3)
exc2_inh_connection = RandomConnection(exc2, inh, connection_rate=0.3, w=torch.ones(exc2.s.numel(), inh.s.numel()) * 0.3)
inh_exc1_connection = RandomConnection(inh, exc1, connection_rate=0.4, w=torch.ones(inh.s.numel(), exc1.s.numel()) * 0.3)
inh_exc2_connection = RandomConnection(inh, exc2, connection_rate=0.4, w=torch.ones(inh.s.numel(), exc2.s.numel()) * 0.3)

# exc_exc_connection = RandomConnection(exc, exc, connection_rate=0.3, w=torch.rand(exc.s.numel(), exc.s.numel()) * 1)
# exc_inh_connection = RandomConnection(exc, inh, connection_rate=0.3, w=torch.rand(exc.s.numel(), inh.s.numel()) * 1)
# inh_exc_connection = RandomConnection(inh, exc, connection_rate=0.3, w=torch.rand(inh.s.numel(), exc.s.numel()) * 1)

pop_current, org_current = populatrion_noisy_step_current((n_pop,), 4000, time, dt, 0, 100)
pop_current[int(time / (1.5 * dt)):int(time / dt) - 1, 0:n_exc1] *= 2
pop_current[:, n_exc1 + n_exc2:n_pop] /= 2
org_current2 = org_current.clone()
org_current3 = org_current.clone()
org_current2[int(time / (1.5 * dt)):int(time / dt) - 1] *= 2
org_current3[:] /= 2

net = Network(learning=False, dt=dt)
net.add_layer(exc1, "exc1")
net.add_layer(exc2, "exc2")
net.add_layer(inh, "inh")
net.add_connection(exc1_exc1_connection, "exc1", "exc1")
net.add_connection(exc2_exc2_connection, "exc2", "exc2")
net.add_connection(exc1_inh_connection, "exc1", "inh")
net.add_connection(exc2_inh_connection, "exc2", "inh")
net.add_connection(inh_exc1_connection, "inh", "exc1")
net.add_connection(inh_exc2_connection, "inh", "exc2")

net.add_monitor(monitor_exc1, "exc1")
net.add_monitor(monitor_exc2, "exc2")
net.add_monitor(monitor_inh, "inh")

inputs = {'exc1': pop_current[:, 0:n_exc1], 'exc2': pop_current[:, n_exc1:n_exc1 + n_exc2],
          'inh': pop_current[:, n_exc1 + n_exc2:n_pop]}

net.run(time, inputs)

s_exc1 = monitor_exc1.get("s")
s_exc2 = monitor_exc2.get("s")
s_inh = monitor_inh.get("s")

plt.rcParams["figure.figsize"] = 20, 10
fig = plt.figure()
axs = fig.subplots(3, 3, gridspec_kw={'hspace': 0.5, 'height_ratios': [3, 1, 1]})
plot_spikes(axs[0, 0], 'Excitatory1 Spikes', s_exc1, dt)
plot_activity(axs[1, 0], 'Excitatory1 Population Activity', s_exc1, dt)
plot_current(axs[2, 0], 'Input Current', org_current2, dt)
plot_spikes(axs[0, 1], 'Excitatory2 Spikes', s_exc2, dt)
plot_activity(axs[1, 1], 'Excitatory2 Population Activity', s_exc2, dt)
plot_current(axs[2, 1], 'Input Current', org_current, dt)
plot_spikes(axs[0, 2], 'Inhibitory Spikes', s_inh, dt)
plot_activity(axs[1, 2], 'Inhibitory Population Activity', s_inh, dt)
plot_current(axs[2, 2], 'Input Current', org_current3, dt)
fig.show()
fig.savefig('rand3_rand13_rand4_1_3_3.png', bbox_inches='tight')
# low rate
#2 both high rate
#3 inh high rate
#4 exc high rate