from cnsproject.network.neural_populations import LIFPopulation, InputPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection, RandomConnection, ConvolutionalConnection, PoolingConnection
from cnsproject.network.network import Network
from cnsproject.encoding.encoders import *
from cnsproject.learning.learning_rules import *
from cnsproject.utils import *
from cnsproject.preprocessing.filter import DoG, Gabor
import matplotlib.pyplot as plt
import torch
import math

time = 25  # time of simulation in ms
dt = 1.0  # dt in ms
image_name = 'image2.png'
image = torch.tensor(plt.imread(image_name))
# image = (image - image.max()) * -1
norm_img = image.clone()
norm_img = norm_img - norm_img.min()
norm_img = norm_img / norm_img.max()
norm_img = norm_img * 255

input_layer = InputPopulation(shape=(1, 512, 512), dt=dt, additive_spike_trace=True, tau_s=100)
conv_layer = LIFPopulation(shape=(1, 256, 256), v_th=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m=50, r_m=10, t_ref=0,
                       init_t_ref=0, dt=dt, additive_spike_trace=True, tau_s=100)
pool_layer = LIFPopulation(shape=(1, 129, 129), v_th=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m=50, r_m=10, t_ref=0,
                       init_t_ref=0, dt=dt, additive_spike_trace=True, tau_s=100)

conv_layer_monitor = Monitor(conv_layer, state_variables=["s"])
pool_layer_monitor = Monitor(pool_layer, state_variables=["s"])

filter = DoG((1, 5), on_center=True).construct(25) * 10
# filter = Gabor(3.0, 6.0, math.pi/2, 0.5, on_center=True).construct(9)

conv_connection = ConvolutionalConnection(input_layer, conv_layer, kernel_size=25, w=filter.unsqueeze(0).unsqueeze(1), stride=2)
pool_connection = PoolingConnection(conv_layer, pool_layer, kernel_size=2, stride=2, trace_scale=10, decay=0.5)

net = Network(learning=True, dt=dt)
net.add_layer(input_layer, "input")
net.add_layer(conv_layer, "conv")
net.add_layer(pool_layer, "pool")

net.add_connection(conv_connection, "input", "conv")
net.add_connection(pool_connection, "conv", "pool")

net.add_monitor(conv_layer_monitor, "conv")
net.add_monitor(pool_layer_monitor, "pool")

enc = PoissonEncoder(time, dt, max_rate=1000)
encoded = enc(norm_img.unsqueeze(0)).float()
inputs = {'input': encoded}

net.run(time, inputs)

conv_s = conv_layer_monitor.get("s")
pool_s = pool_layer_monitor.get("s")

plt.rcParams["figure.figsize"] = 20, 10
fig = plt.figure()
gs_left = fig.add_gridspec(2, 5, wspace=0.1, hspace=0.1, left=0.02, right=0.66)
gs_right = fig.add_gridspec(2, 2, wspace=0.1, hspace=0.1, left=0.74, right=0.98)
axs0_left = list()
axs1_left = list()
axs0_right = list()
axs1_right = list()
for i in range(5):
    axs0_left.append(fig.add_subplot(gs_left[0, i]))
    axs0_left[i].set_yticks([])
    axs0_left[i].set_xticks([])
    axs1_left.append(fig.add_subplot(gs_left[1, i]))
    axs1_left[i].set_yticks([])
    axs1_left[i].set_xticks([])
for i in range(2):
    axs0_right.append(fig.add_subplot(gs_right[0, i]))
    axs0_right[i].set_yticks([])
    axs0_right[i].set_xticks([])
    axs1_right.append(fig.add_subplot(gs_right[1, i]))
    axs1_right[i].set_yticks([])
    axs1_right[i].set_xticks([])

axs0_left[0].imshow(conv_s[0].squeeze(), cmap='gray')
axs0_left[0].set_title('Convolution $t = 0$')
axs0_left[1].imshow(conv_s[int((time / dt) * 0.25)].squeeze(), cmap='gray')
axs0_left[1].set_title('Convolution $t = T/4$')
axs0_left[2].imshow(conv_s[int((time / dt) * 0.5)].squeeze(), cmap='gray')
axs0_left[2].set_title('Convolution $t = T/2$')
axs0_left[3].imshow(conv_s[int((time / dt) * 0.75)].squeeze(), cmap='gray')
axs0_left[3].set_title('Convolution $t = 3T/4$')
axs0_left[4].imshow(conv_s[int(time / dt) - 1].squeeze(), cmap='gray')
axs0_left[4].set_title('Convolution $t = T-1$')

axs1_left[0].imshow(pool_s[0].squeeze(), cmap='gray')
axs1_left[0].set_title('Pooling $t = 0$')
axs1_left[1].imshow(pool_s[int((time / dt) * 0.25)].squeeze(), cmap='gray')
axs1_left[1].set_title('Pooling $t = T/4$')
axs1_left[2].imshow(pool_s[int((time / dt) * 0.5)].squeeze(), cmap='gray')
axs1_left[2].set_title('Pooling $t = T/2$')
axs1_left[3].imshow(pool_s[int((time / dt) * 0.75)].squeeze(), cmap='gray')
axs1_left[3].set_title('Pooling $t = 3T/4$')
axs1_left[4].imshow(pool_s[int(time / dt) - 1].squeeze(), cmap='gray')
axs1_left[4].set_title('Pooling $t = T-1$')

axs0_right[0].imshow(filter, cmap='gray')
axs0_right[0].set_title('Filter')
axs0_right[1].imshow(encoded[1].squeeze(), cmap='gray')
axs0_right[1].set_title('Encoding $t = 0$')

axs1_right[0].imshow(norm_img, cmap='gray')
axs1_right[0].set_title('Stimulus Image')
axs1_right[1].imshow(encoded.sum(0).squeeze(), cmap='gray')
axs1_right[1].set_title('Sum of encodings')

fig.show()
fig.savefig('dog_25_1_2_' + image_name, bbox_inches='tight')
