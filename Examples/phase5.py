from cnsproject.encoding.encoders import *
import matplotlib.pyplot as plt
import torch
from cnsproject.plotting.plotting import plot_spikes


time = 10  # msec
dt = 1  # msec
img = plt.imread('img.png') * 255

# enc = Time2FirstSpikeEncoder(time, dt)
# encoded = enc(torch.tensor(img.tolist(), dtype=torch.float))

enc = PoissonEncoder(time, dt, max_rate=50)
encoded = enc(torch.tensor(img.tolist(), dtype=torch.float))

plt.rcParams["figure.figsize"] = 20, 8
fig = plt.figure()
gs = fig.add_gridspec(2, 7, wspace=0.1, hspace=0.1)
axs0 = list()
for i in range(7):
    axs0.append(fig.add_subplot(gs[0, i]))
    axs0[i].set_yticks([])
    axs0[i].set_xticks([])
axs1 = fig.add_subplot(gs[1, :])
axs0[0].imshow(encoded[0], cmap='gray')
axs0[0].set_title('Encoding $t = 0$')
axs0[1].imshow(encoded[int((time / dt) * 0.25)], cmap='gray')
axs0[1].set_title('Encoding $t = T/4$')
axs0[2].imshow(encoded[int((time / dt) * 0.5)], cmap='gray')
axs0[2].set_title('Encoding $t = T/2$')
axs0[3].imshow(encoded[int((time / dt) * 0.75)], cmap='gray')
axs0[3].set_title('Encoding $t = 3T/4$')
axs0[4].imshow(encoded[int(time / dt) - 1], cmap='gray')
axs0[4].set_title('Encoding $t = T-1$')
axs0[5].imshow(encoded.sum(0), cmap='gray')
axs0[5].set_title('Sum of encodings')
axs0[6].imshow(img, cmap='gray')
axs0[6].set_title('Stimulus Image')
plot_spikes(axs1, 'Encoding Raster Plot', torch.reshape(encoded, (encoded.shape[0], encoded.shape[1] * encoded.shape[2])), 1)
fig.show()
# fig.savefig('.png', bbox_inches='tight')


