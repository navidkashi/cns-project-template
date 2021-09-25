from cnsproject.network.neural_populations import LIFPopulation, ELIFPopulation, AELIFPopulation
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
1- step_current(t * dt, 2000, 10, 90)
2- multi_step_current(t * dt, [2000, 4000, 9000], [10, 50, 70], [30, 65, 93])
3- sine_wave_current(t * dt, 3000, 10, 90)
4- sawtooth_current(t * dt, time, dt, 10000, 10, 90)
'''

is_noisy = False
noise_mean = 0
noise_std = 5000
noise_update_rate = 10
noise_update = noise_update_rate
noise = 0.0

plt.rcParams["figure.figsize"] = 10, 16
fig = plt.figure()
axs = fig.subplots(4, gridspec_kw={'hspace': 0.5})

current = None
neuron = None

neuron1 = LIFPopulation(shape=(1,), v_th=0.0, v_rest=-70.0, v_reset=-75.0, tau_m=8, r_m=10, t_ref=5,
                        init_t_ref=0, dt=dt)
neuron1.reset_state_variables()
neuron2 = ELIFPopulation(shape=(1,), v_th=0.0, v_rest=-70.0, v_reset=-75.0, tau_m=8, r_m=10, t_ref=5,
                         v_rh=-30.0, delta=10, init_t_ref=0, dt=dt)
neuron2.reset_state_variables()
neuron3 = AELIFPopulation(shape=(1,), v_th=0.0, v_rest=-70.0, v_reset=-75.0, tau_m=8, r_m=10, t_ref=5,
                          v_rh=-30.0, delta=10, a=1, b=1000.0, tau_w=50, init_t_ref=0, dt=dt)
neuron3.reset_state_variables()
monitor1 = Monitor(neuron1, state_variables=["s", "v", "current"])
monitor1.reset_state_variables()
monitor2 = Monitor(neuron2, state_variables=["s", "v"])
monitor2.reset_state_variables()
monitor3 = Monitor(neuron3, state_variables=["s", "v", "w_adp"])
monitor3.reset_state_variables()

for t in range(0, int(time / dt)):
    if is_noisy:
        if noise_update <= 0:
            noise = noise_current(t * dt, noise_mean, noise_std, 10, 90)
            noise_update = noise_update_rate
        noise_update -= 1
    else:
        noise = 0.0
    neuron1.forward(torch.tensor(step_current(t * dt, 6000, 10, 90) + noise))
    monitor1.record()
    neuron2.forward(torch.tensor(step_current(t * dt, 6000, 10, 90) + noise))
    monitor2.record()
    neuron3.forward(torch.tensor(step_current(t * dt, 6000, 10, 90) + noise))
    monitor3.record()
s1 = monitor1.get("s")
v1 = monitor1.get("v")
s2 = monitor2.get("s")
v2 = monitor2.get("v")
s3 = monitor3.get("s")
v3 = monitor3.get("v")
w = monitor3.get("w_adp")
current = monitor1.get("current")
# plot_potential(axs[0],
#                'Leaky Integrate and Fire',
#                v1, dt)
# plot_potential(axs[1],
#                'Exponential Leaky Integrate and Fire',
#                v2, dt)
# plot_potential(axs[2],
#                'Adaptive Exponential Leaky Integrate and Fire',
#                v3, dt)
# plot_adaptation(axs[3],
#                 'Adaptive Exponential Leaky Integrate and Fire',
#                 w, dt)
# plot_current(axs[4], 'Input Current', current, dt)
plot_fi_curve(axs[0], 'Leaky Integrate and Fire',
              neuron=neuron1, current_range=[2000, 20000, 500], max_run_time=200, dt=dt)
plot_fi_curve(axs[1], 'Exponential Leaky Integrate and Fire',
              neuron=neuron2, current_range=[2000, 20000, 500], max_run_time=200, dt=dt)
plot_fi_curve(axs[2], 'Adaptive Exponential Leaky Integrate and Fire',
              neuron=neuron3, current_range=[2000, 20000, 500], max_run_time=200, dt=dt)
plot_params(axs[3], neuron3, time, dt)

fig.show()
fig.savefig('lif_elif_aelif_fi.png', bbox_inches='tight')
