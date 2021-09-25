from cnsproject.encoding.encoders import *
import matplotlib.pyplot as plt
import torch
from cnsproject.plotting.plotting import plot_spikes, plot_position_enc

# img = [[3, 0, 7], [4, 9, 1], [8, 2, 10]]
img = plt.imread('img.png') * 256
time = 100  # msec
dt = 1  # msec
size = 50

enc = PositionEncoder(time, dt, mu_start=0.0, mu_step=5.0, sigma=10.0, size=size)
encoded = enc(torch.tensor(img, dtype=torch.float))

plt.rcParams["figure.figsize"] = 10, 5
fig = plt.figure()
axs = fig.subplots()
# plot_position_enc(axs[0], 'Gaussian Functions', enc, torch.tensor(img, dtype=torch.float), time, dt)
plot_spikes(axs, 'Encoding Raster Plot', encoded, dt)

fig.show()
fig.savefig('test.png', bbox_inches='tight')
