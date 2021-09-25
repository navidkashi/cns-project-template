import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from ..network.monitors import Monitor
from ..utils import butter_lowpass_filter
from cnsproject.utils import gaussian
"""
Module for visualization and plotting.

TODO.

Implement this module in any way you are comfortable with. You are free to use\
any visualization library you like. Providing live plotting and animations is\
also a bonus. The visualizations you will definitely need are as follows:

1. F-I curve.
2. Voltage/current dynamic through time.
3. Raster plot of spikes in a neural population.
4. Convolutional weight demonstration.
5. Weight change through time.
"""


def plot_fi_curve(ax, title, neuron, current_range, max_run_time, dt):
    """
        Function for F_I Curve plotting. Note that this function automatically run neuron for defined current range.

        Arguments
        ---------
        ax: plot ax instance.
        title: String.
            The title of plot.
        neuron: NeuralPopulation.
        current_range: list.
            This list contains the range of simulation currents. eg. [start current, stop current, step].
        max_run_time: int.
            Maximum simulation time.
        dt: float.
    """
    neuron.reset_state_variables()
    monitor = Monitor(neuron, state_variables=["s"])
    monitor.reset_state_variables()
    f_tensor = torch.empty(0)
    i_tensor = torch.empty(0)
    for current in torch.arange(current_range[0], current_range[1], current_range[2]):
        for t in range(0, int(max_run_time / dt)):
            neuron.forward(current)
            monitor.record()
        n = torch.nonzero(monitor.get("s"), as_tuple=True)[0]
        periods = list()
        if n.size[0] > 1:
            for i in range(n.size[0] - 1):
                periods.append((n[i + 1] - n[i]) * dt)
            frq = 1.0 / (sum(periods) / len(periods))
        else:
            frq = 0.0
        f_tensor = torch.cat((f_tensor, torch.tensor([frq])))
        i_tensor = torch.cat((i_tensor, torch.tensor([current])))
    ax.plot(i_tensor, f_tensor, 'tab:green')
    ax.set_title(title, fontsize=12)
    ax.set(xlabel='Input Current (pA)', ylabel='Frequency (KHz)')


def plot_potential(ax, title, v, dt):
    """
        Function for membrane potential plotting.

        Arguments
        ---------
        ax: plot ax instance.
        title: String.
            The title of plot.
        v: torch.Tensor.
            Membrane potential tensor.
        dt: float.
    """
    t = torch.arange(0, v.shape[0] * dt, dt)
    ax.plot(t, v)
    ax.set_title(title, fontsize=12)
    ax.set(xlabel='Time (ms)', ylabel='Membrane\nPotential (mV)')


def plot_traces(ax, title, trace, dt):
    """
        Function for spike traces plotting.

        Arguments
        ---------
        ax: plot ax instance.
        title: String.
            The title of plot.
        v: torch.Tensor.
            Membrane potential tensor.
        dt: float.
    """
    time = int(trace.shape[0] * dt)
    t = torch.arange(0, trace.shape[0] * dt, dt)
    ax.plot(t, trace)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-1, time + 1)
    ax.set(xlabel='Time (ms)', ylabel='Spike Trace')

def plot_weights(ax, title, weights, dt):
    """
        Function for spike traces plotting.

        Arguments
        ---------
        ax: plot ax instance.
        title: String.
            The title of plot.
        v: torch.Tensor.
            Membrane potential tensor.
        dt: float.
    """
    time = int(weights.shape[0] * dt)
    t = torch.arange(0, weights.shape[0] * dt, dt)
    ax.plot(t, weights)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-1, time + 1)
    ax.set(xlabel='Time (ms)', ylabel='Weight')


def plot_adaptation(ax, title, w, dt):
    """
        Function for membrane potential plotting.

        Arguments
        ---------
        ax: plot ax instance.
        title: String.
            The title of plot.
        w: torch.Tensor.
            adaptation current tensor.
        dt: float.
    """
    t = torch.arange(0, w.size[0] * dt, dt)
    ax.plot(t, w)
    ax.set_title(title, fontsize=12)
    ax.set(xlabel='Time (ms)', ylabel='Adaptation\nCurrent (pA)')



def plot_current(ax, title, current, dt):
    """
        Function for input current plotting.

        Arguments
        ---------
        ax: plot ax instance.
        title: String.
            The title of plot.
        current: torch.Tensor.
            Input current tensor.
        dt: float.
    """
    time = int(current.size[0] * dt)
    t = torch.arange(0, time, dt)
    ax.plot(t, current, 'tab:orange')
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(0, time, int(time / 5)))
    ax.set_xlim(0, time)
    ax.set_ylim(0, 12000)
    ax.set(xlabel='Time (ms)', ylabel='Input\nCurrent (pA)')


def plot_activity(ax, title, s, dt):
    """
        Function for population activity plotting.

        Arguments
        ---------
        ax: plot ax instance.
        title: String.
            The title of plot.
        s: torch.Tensor.
            Population spike tensor.
        dt: float.
    """
    order = 1
    fs = 100.0  # sample rate, Hz
    cutoff = 10  # desired cutoff frequency of the filter, Hz

    time = int(s.shape[0] * dt)
    t = range(0, time)
    activity_dt = torch.sum(s.float(), dim=1)
    activity = torch.zeros((len(t),))
    for i, val in enumerate(activity_dt):
        activity[int(i * dt)] += val
    activity /= s.size[1]
    activity *= dt
    y = butter_lowpass_filter(activity, cutoff, fs, order)
    ax.plot(t, y, 'tab:red')
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(0, time, int(time / 5)))
    ax.set_xlim(0, time)
    ax.set_ylim(0, 0.005)
    ax.set(xlabel='Time (ms)', ylabel='Population\nActivity')


def plot_value(ax, title, val, dt):
    """
        Function for population activity plotting.

        Arguments
        ---------
        ax: plot ax instance.
        title: String.
            The title of plot.
        s: torch.Tensor.
            Population spike tensor.
        dt: float.
    """

    time = int(val.shape[0] * dt)
    t = range(0, time)

    ax.plot(t, val)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(0, time, int(time / 5)))
    ax.set_xlim(-1, time + 1)
    ax.set(xlabel='Time (ms)', ylabel='Value')

def plot_spikes(ax, title, s, dt, base_index=0):
    """
        Function for spike raster plotting.

        Arguments
        ---------
        ax: plot ax instance.
        s: torch.Tensor.
            Spike tensor.
        dt: float.
    """
    n = torch.nonzero(torch.t(s))
    time = int(s.shape[0] * dt)
    ax.scatter(n[:, 1] * dt, n[:, 0] + base_index, marker='.')
    ax.set_title(title, fontsize=16)
    ax.set_xticks(range(0, time, int(time / 5)))
    ax.set_xlim(-1, time+1)
    ax.set(xlabel='Time (ms)', ylabel='Neuron Index')


def plot_position_enc(ax, title, enc, data, time, dt, base_index=0):
    """
        Function for spike raster plotting.

        Arguments
        ---------
        ax: plot ax instance.
        s: torch.Tensor.
            Spike tensor.
        dt: float.
    """
    start = enc.mu_start - enc.sigma * 3
    end = enc.mu_start + enc.mu_step * enc.size + enc.sigma * 3
    x = torch.arange(start, end, .1)
    for i in range(enc.size):
        ax.plot(time - gaussian(x, enc.mu_start + enc.mu_step * i, enc.sigma) * time, x, label='{:d}'.format(i + 1))
    for i in data.flatten():
        ax.plot([-1, time + 1], [i, i], ls=':', label='{:d}'.format(int(i.item())))
    ax.plot([time * enc.time_th, time * enc.time_th], [start, end], color='black')
    ax.set_xticks(range(0, time, int(time / 5)))
    ax.set_xlim(-1, time + 1)
    ax.legend(loc='best', ncol=3, fancybox=True, shadow=True)
    ax.set_title(title, fontsize=16)

def plot_params(ax, neuron, time, dt):
    """
        Function for plotting the parameters of neuron.

        Arguments
        ---------
        ax: plot ax instance.
        neuron: NeuralPopulation.
        time: int.
            Simulation time.
        dt: float.
    """
    params_t = 'Total Time Frame: ' + str(time) + ' ms'
    params_t += '\ndt: ' + str(dt) + ' ms'
    params_t += '\n\nInitial Refractory Time: ' + str(neuron.__dict__['init_t_ref']) + ' ms'
    params_t += '\nRefractory Period: ' + str(neuron.__dict__['t_ref']) + ' ms'
    params_t += '\n$\\tau_w$: ' + str(neuron.__dict__['tau_w']) + ' ms'
    params_t += '\n$\\Delta_T$: ' + str(neuron.__dict__['delta']) + ' mV'
    params_t += '\n$a$: ' + str(neuron.__dict__['a'])
    params_m = '$R_m$: ' + str(neuron.__dict__['r_m']) + ' M\u03A9'
    params_m += '\n$\u03C4_m$: ' + str(neuron.__dict__['tau_m']) + ' ms'
    params_m += '\n$V_{threshold}$: ' + str(neuron.__dict__['v_th']) + ' mV'
    params_m += '\n$V_{rest}$: ' + str(neuron.__dict__['v_rest']) + ' mV'
    params_m += '\n$V_{reset}$: ' + str(neuron.__dict__['v_reset']) + ' mV'
    params_m += '\n$\\theta_{rh}$: ' + str(neuron.__dict__['v_rh']) + ' mV'
    params_m += '\n$b$: ' + str(neuron.__dict__['b']) + ' pA'
    ax.text(0.3, 0.4, params_t, size=14,
            va="center", ha="center", multialignment="left")
    ax.text(0.7, 0.4, params_m, size=14,
            va="center", ha="center", multialignment="left")
    ax.set_title('Parameters', fontsize=16)
    ax.axis('off')


def plot_default_params(ax, neuron, time, dt):
    """
        Function for plotting the default parameters of neuron. Note that this function only used for plotting default
        parameters in plots of report.

        Arguments
        ---------
        ax: plot ax instance.
        neuron: NeuralPopulation.
        time: int.
            Simulation time.
        dt: float.

    """
    params_t = 'Total Time Frame: ' + str(time) + ' ms'
    params_t += '\ndt: ' + str(dt) + ' ms'
    params_t += '\nInitial Refractory Time: ' + str(neuron.__dict__['init_t_ref']) + ' ms'
    params_t += '\nRefractory Period: ' + str(neuron.__dict__['t_ref']) + ' ms'
    params_t += '\n$\\Delta_T$: ' + str(neuron.__dict__['delta']) + ' mV'
    params_m = '$R_m$: ' + str(neuron.__dict__['r_m']) + ' M\u03A9'
    params_m += '\n$\u03C4_m$: ' + str(neuron.__dict__['tau_m']) + ' ms'
    params_m += '\n$V_{rest}$: ' + str(neuron.__dict__['v_rest']) + ' mV'
    params_m += '\n$V_{reset}$: ' + str(neuron.__dict__['v_reset']) + ' mV'
    params_m += '\n$\\theta_{rh}$: ' + str(neuron.__dict__['v_rh']) + ' mV'
    ax.text(0.3, 0.55, params_t, size=16,
            va="center", ha="center", multialignment="left")
    ax.text(0.7, 0.5, params_m, size=16,
            va="center", ha="center", multialignment="left")
    ax.set_title('Parameters', fontsize=16)
    ax.axis('off')


def update_step(t, dt, neuron_population, monitor, potential_plot):
    '''
    TODO.
    Frame update function for animation plot.

    '''

    curr = torch.tensor(3000)
    neuron_population.forward(curr)
    monitor.record()
    v = monitor.get('v')
    potential_plot.set_xdata(t * dt)
    print(t * dt)
    potential_plot.set_ydata(1)  # update the data.
    return potential_plot,


def plot_population_animation(fig, title, neuron_population, time, dt, potential=True, current=False, spike=False,
                              params=True):
    '''
    TODO.

    Function for animation plotting.

    '''
    number_of_subplots = potential + current + spike + params
    state_variables = []
    if potential:
        state_variables.append('v')
    if current:
        state_variables.append('current')
    if spike:
        state_variables.append('spike')
    monitor = Monitor(neuron_population, state_variables=state_variables)
    neuron_population.reset_state_variables()
    monitor.reset_state_variables()
    axs = fig.subplots(number_of_subplots, gridspec_kw={'hspace': 0.5})
    # for i in range(number_of_subplots):
    #     if potential:
    #         plot_potential(axs[i], )
    t = torch.arange(0, time, dt)
    potential_plot, = axs[0].plot([], [])
    ani = FuncAnimation(
        fig, update_step, fargs=(dt, neuron_population, monitor, potential_plot), interval=int(time / dt), blit=False)
    return fig
