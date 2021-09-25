from cnsproject.network.neural_populations import LIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.utils import *
import matplotlib.pyplot as plt
import torch

from cnsproject.plotting.plotting import plot_potential, plot_fi_curve, plot_current, plot_spikes, plot_params, \
    plot_default_params

time = 100  # time of simulation in ms
dt = 0.03125  # dt in ms

'''
Demo currents
1- step_current(t * dt, 3000, 10, 90)
2- multi_step_current(t * dt, [2000, 4000, 9000], [10, 50, 70], [30, 65, 93])
3- sine_wave_current(t * dt, 3000, 10, 90)
'''

is_noisy = True
noise_mean = 0
noise_std = 3000
noise_update_rate = 10
noise_update = noise_update_rate
noise = 0.0

v_th = [-50.0, 0.0, -50.0, -50.0, -50.0]
tau_m = [8, 8, 32, 8, 8]
r_m = [10, 10, 10, 20, 10]
t_ref = [0, 0, 0, 0, 5]

plt.rcParams["figure.figsize"] = 10, 18
fig = plt.figure()
axs = fig.subplots(7, gridspec_kw={'hspace': 0.5})

current = None
neuron = None

for i, (v_th, tau_m, r_m, t_ref) in enumerate(zip(v_th, tau_m, r_m, t_ref)):
    neuron = LIFPopulation(shape=(1,), v_th=v_th, v_rest=-70.0, v_reset=-75.0, tau_m=tau_m, r_m=r_m, t_ref=t_ref,
                           init_t_ref=0, dt=dt)
    neuron.reset_state_variables()
    monitor = Monitor(neuron, state_variables=["s", "v", "current"])
    monitor.reset_state_variables()

    for t in range(0, int(time / dt)):
        if is_noisy:
            if noise_update <= 0:
                noise = noise_current(t * dt, noise_mean, noise_std, 10, 90)
                noise_update = noise_update_rate
            noise_update -= 1
        else:
            noise = 0.0
        neuron.forward(torch.tensor(sawtooth_current(t * dt, time, dt, 10000, 10, 90) + noise))
        monitor.record()
    s = monitor.get("s")
    v = monitor.get("v")
    current = monitor.get("current")

    plot_potential(axs[i],
                   '$V_{threshold}$:' +
                   ' {:.1f} mV, $R_m$: {:d} M\u03A9, $\u03C4_m$: {:d} ms, Refractory Period: {:d} ms'.format(
                       v_th,
                       r_m,
                       tau_m,
                       t_ref),
                   v, dt)

plot_current(axs[5], 'Input Current', current, dt)
plot_default_params(axs[6], neuron, time, dt)

fig.show()
fig.savefig('test.png', bbox_inches='tight')