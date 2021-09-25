"""
Module for utility functions.

Use this module to implement any required utility function.

Note: You are going to need to implement DoG and Gabor filters. A possible opt
ion would be to write them in this file but it is not a must and you can define\
a separate module/package for them.
"""
from math import sin

import numpy as np
import torch
from numpy import random
from scipy import signal
from scipy.signal import butter, lfilter
import os

def step_current(t, amplitude, start_point, stop_point) -> float:
    """
    Step current generator that depends on time.

    Arguments
    ---------
    t: float.
        time.
    amplitude: float.
        Amplitude of the current.
    start_point: int.
        Start time of the step current.
    stop_point: int.
        Stop time of the step current.

    Returns
    -------
    Current value.
    """
    if start_point <= t < stop_point:
        return amplitude
    else:
        return 0.0


def multi_step_current(t, amplitudes, start_points, stop_points) -> float:
    """
    Multi Step current generator that depends on time. This function create multi steps according to the start points and
    stop points lists.

    Arguments
    ---------
    t: float.
        time.
    amplitudes: list of float.
        List that contains amplitudes of every step function.
    start_points: list of int.
        List that contains start points of every step function.
    stop_points: list of int.
        List that contains stop points of every step function.

    Returns
    -------
    Current value.
    """
    for start, stop, amplitude in zip(start_points, stop_points, amplitudes):
        if start <= t < stop:
            return float(amplitude)
    return 0.0


def sine_wave_current(t, amplitude, start_point, stop_point) -> float:
    """
    Sine wave current generator that depends on time.

    Arguments
    ---------
    t: float.
        time.
    amplitude: float.
        Amplitude of the current.
    start_point: int.
        Start time of the step current.
    stop_point: int.
        Stop time of the step current.

    Returns
    -------
    Current value.
    """
    if start_point <= t < stop_point:
        return amplitude * (sin(t) + 1)
    else:
        return 0.0


def sawtooth_current(t, time, dt, amplitude, start_point, stop_point) -> float:
    """
    Sawtooth current generator that depends on time.

    Arguments
    ---------
    t: float.
        time.
    amplitude: float.
        Amplitude of the current.
    start_point: int.
        Start time of the step current.
    stop_point: int.
        Stop time of the step current.

    Returns
    -------
    Current value.
    """
    time_frame = np.arange(0, time / dt, 1)
    if start_point <= t < stop_point:
        return ((signal.sawtooth(time_frame / 100) + 0.8) * amplitude)[int(t / dt)]
    else:
        return 0.0


def noise_current(t, mean, std, start_point, stop_point):
    """
    Generate a random noise value in defined time frame.

    Arguments
    ---------
    t: float.
        time.
    mean: float.
        mean value of generate noise.
    std: float.
        Standard deviation of generated noise. Must be non negative.
    start_point: int.
        Start time of the noise generation.
    stop_point: int.
        Stop time of the noise generation.

    Returns
    -------
    Random noise value.
    """
    if start_point <= t < stop_point:
        return abs(random.normal(mean, std))
    else:
        return 0.0


def populatrion_noisy_step_current(shape, amplitude, time, dt, start_point, stop_point):
    """
    Generate a random noise value in defined time frame.

    Arguments
    ---------
    t: float.
        time.
    mean: float.
        mean value of generate noise.
    std: float.
        Standard deviation of generated noise. Must be non negative.
    start_point: int.
        Start time of the noise generation.
    stop_point: int.
        Stop time of the noise generation.

    Returns
    -------
    Random noise value.
    """
    order = 3
    fs = 100.0  # sample rate, Hz
    cutoff = 1  # desired cutoff frequency of the filter, Hz

    current = list()
    curr = torch.zeros((int(time / dt),) + shape)
    for t in range(0, int(time / dt)):
        if start_point <= t * dt < stop_point:
            noise = random.normal(0, 1) * 2000
            current.append(amplitude + noise)
        else:
            current.append(.0)

    y = butter_lowpass_filter(current, cutoff, fs, order)
    for i in range(shape[0]):
        if abs(random.random()) > 0.7:
            curr[:, i] = torch.tensor(y) + random.normal(0, 2000)
        else:
            curr[:, i] = torch.tensor(y) + random.normal(0, 600)
    return curr, torch.tensor(y)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def gaussian(x: torch.Tensor, mu: float, sig: float) -> torch.Tensor:
    return torch.exp(-torch.pow((x - mu) / sig, 2.) / 2)


def sign(a):
    return (a > 0) - (a < 0)


