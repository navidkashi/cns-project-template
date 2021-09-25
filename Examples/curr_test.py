from cnsproject.network.neural_populations import LIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection
from cnsproject.utils import *
from cnsproject.encoding.encoders import *
import matplotlib.pyplot as plt
import torch
import numpy as np

from cnsproject.plotting.plotting import plot_potential, plot_fi_curve, plot_current, plot_spikes, plot_params, \
    plot_default_params
from cnsproject.utils import gaussian
img = plt.imread('test1.jpg').astype(float)

test = [[1, 2], [0, 1]]
enc = PoissonEncoder(10, 1)
enc1 = PositionEncoder(10, 1)
encoded = enc(torch.tensor(img.tolist(), dtype=torch.float))
# encoded = enc1(torch.tensor(img.tolist(), dtype=torch.float))
plot_spikes()
plt.imshow(encoded.sum(0), cmap='gray')
plt.show()
