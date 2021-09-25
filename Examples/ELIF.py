from cnsproject.network.neural_populations import ELIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.utils import *
import matplotlib.pyplot as plt
import torch

from cnsproject.plotting.plotting import plot_potential, plot_fi_curve, plot_current, plot_spikes, plot_params, \
    plot_default_params

time = 100  # time of simulation in ms
dt = 0.03125  # dt in ms

neuron = ELIFPopulation(shape=(1,), v_th=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m=8, r_m=10, t_ref=5, delta=1,
                        v_rh=-45.0, init_t_ref=0, dt=dt)
neuron.reset_state_variables()
monitor = Monitor(neuron, state_variables=["s", "v", "current"])
monitor.reset_state_variables()

for t in range(0, int(time / dt)):
    neuron.forward(torch.tensor(step_current(t * dt, 3000, 10, 90)))
    monitor.record()
s = monitor.get("s")
v = monitor.get("v")
current = monitor.get("current")

plt.rcParams["figure.figsize"] = 10, 12
fig = plt.figure()
axs = fig.subplots(4, gridspec_kw={'hspace': 0.5})
plot_potential(axs[0], 'Exponential Leaky Integrate and Fire', v, dt)
plot_current(axs[1], 'Input Current', current, dt)
# plot_fi_curve(axs[2], 'F_I Curve', neuron=neuron, current_range=[1000, 5000, 100], max_run_time=150, dt=dt)
plot_params(axs[3], neuron, time, dt)
fig.show()
