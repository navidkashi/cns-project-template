from cnsproject.network.neural_populations import AELIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.utils import *
import matplotlib.pyplot as plt
import torch

from cnsproject.plotting.plotting import plot_potential, plot_fi_curve, plot_current, plot_spikes, plot_params, \
    plot_default_params

time = 100  # time of simulation in ms
dt = 0.03125  # dt in ms

v_th = [0.0, 0.0, 0.0, 0.0, 0.0]
a = [1, 200, 1, 1, 1]
b = [500.0, 500.0, 1000.0, 500.0, 500.0]
tau_w = [100, 100, 100, 50, 10]

plt.rcParams["figure.figsize"] = 10, 18
fig = plt.figure()
axs = fig.subplots(6, gridspec_kw={'hspace': 0.5})

current = None
neuron = None

for i, (v_th, a, b, tau_w) in enumerate(zip(v_th, a, b, tau_w)):
    neuron = AELIFPopulation(shape=(1,), v_th=v_th, v_rest=-70.0, v_reset=-75.0, tau_m=8, r_m=10, t_ref=3, delta=10,
                             v_rh=-30.0, a=a, b=b, tau_w=tau_w, init_t_ref=0, dt=dt)
    neuron.reset_state_variables()
    monitor = Monitor(neuron, state_variables=["s", "v", "current"])
    monitor.reset_state_variables()

    plot_fi_curve(axs[i],
                  '$V_{threshold}$:' +
                  ' {:.1f} mV, a: {:d}, b: {:.1f} pA, $\\tau_w$: {:d} '
                  'ms'.format(
                      v_th,
                      a,
                      b,
                      tau_w),
                  neuron=neuron, current_range=[1000, 20000, 200], max_run_time=200, dt=dt)

plot_default_params(axs[5], neuron, time, dt)

fig.show()
fig.savefig('test.png', bbox_inches='tight')
