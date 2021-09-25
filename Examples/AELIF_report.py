from cnsproject.network.neural_populations import AELIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.utils import *
import matplotlib.pyplot as plt
import torch

from cnsproject.plotting.plotting import plot_potential, plot_fi_curve, plot_current, plot_spikes, plot_params, \
    plot_default_params, plot_adaptation

time = 100  # time of simulation in ms
dt = 0.03125  # dt in ms

'''
Demo currents
1- step_current(t * dt, 8000, 10, 90)
2- multi_step_current(t * dt, [6000, 10000, 14000], [10, 50, 70], [30, 65, 93])
3- sine_wave_current(t * dt, 7000, 10, 90)
4- sawtooth_current(t * dt, time, dt, 15000, 10, 90)
'''

is_noisy = False
noise_mean = 0
noise_std = 4000
noise_update_rate = 10
noise_update = noise_update_rate
noise = 0.0

v_th = [0.0, 0.0, 0.0, 0.0, 0.0]
a = [1, 200, 1, 1, 1]
b = [500.0, 500.0, 1000.0, 500.0, 500.0]
tau_w = [100, 100, 100, 50, 10]

plt.rcParams["figure.figsize"] = 10, 18
fig = plt.figure()
axs = fig.subplots(7, gridspec_kw={'hspace': 0.5})

current = None
neuron = None

for i, (v_th, a, b, tau_w) in enumerate(zip(v_th, a, b, tau_w)):
    neuron = AELIFPopulation(shape=(1,), v_th=v_th, v_rest=-70.0, v_reset=-75.0, tau_m=8, r_m=10, t_ref=3, delta=10,
                             v_rh=-30.0, a=a, b=b, tau_w=tau_w, init_t_ref=0, dt=dt)
    neuron.reset_state_variables()
    monitor = Monitor(neuron, state_variables=["s", "v", "current", "w_adp"])
    monitor.reset_state_variables()

    for t in range(0, int(time / dt)):
        if is_noisy:
            if noise_update <= 0:
                noise = noise_current(t * dt, noise_mean, noise_std, 10, 90)
                noise_update = noise_update_rate
            noise_update -= 1
        else:
            noise = 0.0
        neuron.forward(torch.tensor(sawtooth_current(t * dt, time, dt, 15000, 10, 90) + noise))
        monitor.record()
    s = monitor.get("s")
    v = monitor.get("v")
    w = monitor.get("w_adp")
    current = monitor.get("current")
    # plot_potential(axs[i],
    #                '$V_{threshold}$:' +
    #                ' {:.1f} mV, a: {:d}, b: {:.1f} pA, $\\tau_w$: {:d} '
    #                'ms'.format(
    #                    v_th,
    #                    a,
    #                    b,
    #                    tau_w),
    #                v, dt)
    plot_adaptation(axs[i],
                   '$V_{threshold}$:' +
                   ' {:.1f} mV, a: {:d}, b: {:.1f} pA, $\\tau_w$: {:d} '
                   'ms'.format(
                       v_th,
                       a,
                       b,
                       tau_w),
                   w, dt)

plot_current(axs[5], 'Input Current', current, dt)
plot_default_params(axs[6], neuron, time, dt)

fig.show()
fig.savefig('aelif_saw_w.png', bbox_inches='tight')
