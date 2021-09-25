from cnsproject.network.neural_populations import LIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection, RandomConnection
from cnsproject.utils import *
import matplotlib.pyplot as plt
import torch
import numpy as np

from cnsproject.plotting.plotting import plot_potential, plot_activity, plot_current, plot_spikes, plot_params, \
    plot_default_params

time = 100  # time of simulation in ms
dt = 0.03125  # dt in ms

n_pop = 1000
n_exc = int(n_pop * 0.8)
n_inh = int(n_pop * 0.2)

exc = LIFPopulation(shape=(n_exc,), v_th=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m=50, r_m=10, t_ref=0,
                       init_t_ref=0, dt=dt)
inh = LIFPopulation(shape=(n_inh,), v_th=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m=25, r_m=10, t_ref=0,
                       init_t_ref=0, dt=dt, is_inhibitory=True)

exc.reset_state_variables()
inh.reset_state_variables()
monitor_exc = Monitor(exc, state_variables=["s"])
monitor_exc.reset_state_variables()
monitor_inh = Monitor(inh, state_variables=["s"])
monitor_inh.reset_state_variables()


# torch.rand(exc.s.numel(), exc.s.numel()) * 1
exc_exc_connection = DenseConnection(exc, exc, w=torch.ones(exc.s.numel(), exc.s.numel()) * 0.1)
exc_inh_connection = DenseConnection(exc, inh, w=torch.ones(exc.s.numel(), inh.s.numel()) * 0.2)
inh_exc_connection = DenseConnection(inh, exc, w=torch.ones(inh.s.numel(), exc.s.numel()) * 0.3)
con = {}
con['exc_to_inh'] = exc_inh_connection
for c in con:
    print(c.split('_to_'))
# exc_exc_connection = DenseConnection(exc, exc, w=torch.rand(exc.s.numel(), exc.s.numel()) * 0.5)
# exc_inh_connection = DenseConnection(exc, inh, w=torch.rand(exc.s.numel(), inh.s.numel()) * 1)
# inh_exc_connection = DenseConnection(inh, exc, w=torch.rand(inh.s.numel(), exc.s.numel()) * 0.5)

# exc_exc_connection = RandomConnection(exc, exc, connection_rate=0.05, w=torch.ones(exc.s.numel(), exc.s.numel()) * 1)
# exc_inh_connection = RandomConnection(exc, inh, connection_rate=0.05, w=torch.ones(exc.s.numel(), inh.s.numel()) * 1)
# inh_exc_connection = RandomConnection(inh, exc, connection_rate=0.05, w=torch.ones(inh.s.numel(), exc.s.numel()) * 1)

# exc_exc_connection = RandomConnection(exc, exc, connection_rate=0.3, w=torch.rand(exc.s.numel(), exc.s.numel()) * 1)
# exc_inh_connection = RandomConnection(exc, inh, connection_rate=0.3, w=torch.rand(exc.s.numel(), inh.s.numel()) * 1)
# inh_exc_connection = RandomConnection(inh, exc, connection_rate=0.3, w=torch.rand(inh.s.numel(), exc.s.numel()) * 1)

pop_current, org_current = populatrion_noisy_step_current((n_pop,), 4000, time, dt, 10, 100)

for t in range(0, int(time / dt)):
    exc.forward(pop_current[t][0:n_exc])
    inh.forward(pop_current[t][n_exc:n_pop])
    exc_input = exc_exc_connection.compute()
    exc_input += inh_exc_connection.compute()
    inh_input = exc_inh_connection.compute()
    # exc.inject_v = exc_input
    # inh.inject_v = inh_input
    monitor_exc.record()
    monitor_inh.record()

s_exc = monitor_exc.get("s")
s_inh = monitor_inh.get("s")

plt.rcParams["figure.figsize"] = 10, 12
fig = plt.figure()
axs = fig.subplots(4, gridspec_kw={'hspace': 0.5, 'height_ratios': [3, 1, 1, 1]})
plot_spikes(axs[0], s_exc, dt)
plot_spikes(axs[0], s_inh, dt, base_index=n_exc)
plot_activity(axs[1], 'Excitatory Population Activity', s_exc, dt)
plot_activity(axs[2], 'Inhibitory Population Activity', s_inh, dt)
plot_current(axs[3], 'Input Current', org_current, dt)
fig.show()
fig.savefig('no.png', bbox_inches='tight')
# low rate
#2 both high rate
#3 inh high rate
#4 exc high rate