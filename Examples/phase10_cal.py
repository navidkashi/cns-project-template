import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets
from torchvision.datasets import ImageFolder

from cnsproject.decision.decision import *
from cnsproject.encoding.encoders import *
from cnsproject.learning.learning_rules import *
from cnsproject.network.monitors import Monitor
from cnsproject.network.network import Network
from cnsproject.network.neural_populations import LIFPopulation, InputPopulation
from cnsproject.preprocessing.convolution import TorchConvolution
from cnsproject.preprocessing.filter import DoG
from cnsproject.utils import *

transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.Resize((154, 154)),
     transforms.ToTensor(), ])
dataset = ImageFolder("dataset/navid", transform)
sample_idx = random.randint(0, len(dataset) - 1)

time = 5  # time of simulation in ms
dt = 1.0  # dt in ms

import numpy as np

plt.style.use('seaborn-white')
plt_idx = 0

# for t in range(6):
#     plt_idx += 1
#     ax = plt.subplot(5, 6, plt_idx)
#     plt.setp(ax, xticklabels=[])
#     plt.setp(ax, yticklabels=[])
#     if t == 0:
#         ax.set_ylabel('Feature ' + str(0))
#     plt.imshow(sw[t].numpy(), cmap='gray')
# plt.show()

input_layer = InputPopulation(shape=(1, 150, 150), dt=dt, additive_spike_trace=False, tau_s=10)
# pool_layer = LIFPopulation(shape=(1, 30, 30), v_th=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m=50, r_m=10, t_ref=3,
#                        init_t_ref=0, dt=dt, additive_spike_trace=True, tau_s=50)

conv_layer = LIFPopulation(shape=(20, 21, 21), v_th=-50.0, v_rest=-70.0, v_reset=-70.0, tau_m=10, r_m=10, t_ref=0,
                           init_t_ref=0, dt=dt, additive_spike_trace=False, tau_s=10)

conv_layer_monitor = Monitor(conv_layer, state_variables=["s"])
# pool_layer_monitor = Monitor(pool_layer, state_variables=["s"])

conv_connection = TorchConvolutionalConnection(input_layer, conv_layer, kernel_size=50, stride=5, padding=False,
                                               learning_rule=FlatSTDP, lr=[0.001, 0.001], weight_decay=0, wmax=0.5,
                                               w=torch.rand(conv_layer.s.shape[0], input_layer.s.shape[0], 50, 50) * 0.1
                                               , window=5)
# pool_connection = TorchPoolingConnection(input_layer, pool_layer, kernel_size=3, stride=2, trace_scale=5, decay=0.2, padding=False)

net = Network(learning=True, dt=dt, decision=WinnerTakeAllDecision, kwta=10, inhibition_radius=10)
net.add_layer(input_layer, "input")
# net.add_layer(pool_layer, "pool")
net.add_layer(conv_layer, "conv")

# net.add_connection(pool_connection, "input", "pool")
net.add_connection(conv_connection, "input", "conv")

net.add_monitor(conv_layer_monitor, "conv")
# net.add_monitor(pool_layer_monitor, "pool")

filter = DoG((1, 5), on_center=True).construct(5)
conv = TorchConvolution()

dataset = datasets.Caltech101(
    root="data",
    download=True,
    transform=transform,

)

idx = torch.tensor(dataset.y) == 1
sub_dset = torch.utils.data.dataset.Subset(dataset, np.where(idx == 1)[0])

indices = list(range(len(sub_dset)))
random.shuffle(indices)
split_point = int(1 * len(indices))
sub_indices = indices[:split_point]
print("Size of the dataset:", len(sub_indices))
data_loader = DataLoader(sub_dset, sampler=SubsetRandomSampler(sub_indices))

for it in range(10):
    print('\rIteration:', it)
    for data, _ in data_loader:
        # print(data.shape)
        # exit()
        image = data.squeeze(0).squeeze(0)
        img = conv.execute(image, filter, padding=False)
        norm_img = img.clone().unsqueeze(0).unsqueeze(0)
        norm_img[norm_img < 0.005] = 0
        # print(norm_img.shape)
        enc = Intensity2Latency(time=time, dt=dt)
        encoded = enc(norm_img).float()
        # print(encoded.shape)
        inputs = {'input': encoded}

        net.run(time, inputs, decision_pop='conv')

print(conv_connection.w.shape)

conv_s = conv_layer_monitor.get("s")
# pool_s = pool_layer_monitor.get("s")

plt.rcParams["figure.figsize"] = 20, 12
fig = plt.figure()
gs_left = fig.add_gridspec(4, 5, wspace=0.1, hspace=0.1, left=0.02, right=0.66)
gs_right = fig.add_gridspec(2, 2, wspace=0.1, hspace=0.1, left=0.74, right=0.98)
axs0_left = list()
axs1_left = list()
axs2_left = list()
axs3_left = list()
axs0_right = list()
axs1_right = list()
for i in range(5):
    axs0_left.append(fig.add_subplot(gs_left[0, i]))
    axs0_left[i].set_yticks([])
    axs0_left[i].set_xticks([])
    axs1_left.append(fig.add_subplot(gs_left[1, i]))
    axs1_left[i].set_yticks([])
    axs1_left[i].set_xticks([])
    axs2_left.append(fig.add_subplot(gs_left[2, i]))
    axs2_left[i].set_yticks([])
    axs2_left[i].set_xticks([])
    axs3_left.append(fig.add_subplot(gs_left[3, i]))
    axs3_left[i].set_yticks([])
    axs3_left[i].set_xticks([])
for i in range(2):
    axs0_right.append(fig.add_subplot(gs_right[0, i]))
    axs0_right[i].set_yticks([])
    axs0_right[i].set_xticks([])
    axs1_right.append(fig.add_subplot(gs_right[1, i]))
    axs1_right[i].set_yticks([])
    axs1_right[i].set_xticks([])

for i in range(5):
    axs0_left[i].imshow(conv_connection.w[i].squeeze(), cmap='gray')
    axs0_left[i].set_title('Kernel ' + str(i))
    axs1_left[i].imshow(conv_connection.w[i + 5].squeeze(), cmap='gray')
    axs1_left[i].set_title('Kernel ' + str(i + 5))
    axs2_left[i].imshow(conv_connection.w[i + 10].squeeze(), cmap='gray')
    axs2_left[i].set_title('Kernel ' + str(i + 10))
    axs3_left[i].imshow(conv_connection.w[i + 15].squeeze(), cmap='gray')
    axs3_left[i].set_title('Kernel ' + str(i + 15))

axs0_right[0].imshow(filter, cmap='gray')
axs0_right[0].set_title('Filter')
axs0_right[1].imshow(encoded[0].squeeze(), cmap='gray')
axs0_right[1].set_title('Encoding $t = 0$')

axs1_right[0].imshow(norm_img.squeeze(0).squeeze(0), cmap='gray')
axs1_right[0].set_title('Stimulus Image')
axs1_right[1].imshow(encoded.sum(0).squeeze(), cmap='gray')
axs1_right[1].set_title('Sum of encodings')

fig.show()

fig.savefig('caltech_10_10', bbox_inches='tight')
