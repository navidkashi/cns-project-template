from cnsproject.network.neural_populations import LIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.utils import *
import matplotlib.pyplot as plt
import torch

from cnsproject.plotting.plotting import plot_potential, plot_fi_curve, plot_current, plot_spikes, plot_params, \
    plot_default_params

time = 100  # time of simulation in ms
dt = 0.03125  # dt in ms

v_th = [-50.0, 0.0, -50.0, -50.0, -50.0]
tau_m = [8, 8, 32, 8, 8]
r_m = [10, 10, 10, 20, 10]
t_ref = [0, 0, 0, 0, 5]

plt.rcParams["figure.figsize"] = 10, 18
fig = plt.figure()
axs = fig.subplots(6, gridspec_kw={'hspace': 0.5})

current = None
neuron = None

for i, (v_th, tau_m, r_m, t_ref) in enumerate(zip(v_th, tau_m, r_m, t_ref)):
    neuron = LIFPopulation(shape=(1,), v_th=v_th, v_rest=-70.0, v_reset=-75.0, tau_m=tau_m, r_m=r_m, t_ref=t_ref,
                           init_t_ref=0, dt=dt)
    neuron.reset_state_variables()
    monitor = Monitor(neuron, state_variables=["s", "v", "current"])
    monitor.reset_state_variables()

    plot_fi_curve(axs[i],
                    '$V_{threshold}$:' +
                    ' {:.1f} mV, $R_m$: {:d} M\u03A9, $\u03C4_m$: {:d} ms, Refractory Period: {:d} ms'.format(
                       v_th,
                       r_m,
                       tau_m,
                       t_ref),
                    neuron=neuron, current_range=[500, 15000, 100], max_run_time=200, dt=dt)

plot_default_params(axs[5], neuron, time, dt)

fig.show()
fig.savefig('test.png', bbox_inches='tight')