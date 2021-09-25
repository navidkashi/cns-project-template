from cnsproject.decision.decision import *
from cnsproject.network.neural_populations import LIFPopulation, InputPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection, RandomConnection, ConvolutionalConnection, PoolingConnection
from cnsproject.network.network import Network
from cnsproject.encoding.encoders import *
from cnsproject.learning.learning_rules import *
from cnsproject.utils import *
from cnsproject.preprocessing.filter import DoG, Gabor
import matplotlib.pyplot as plt
from cnsproject.preprocessing.filter import DoG, Gabor
from cnsproject.preprocessing.convolution import Convolution
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(), ])
dataset = ImageFolder("dataset/eth", transform)
sample_idx = random.randint(0, len(dataset) - 1)

filter = DoG((1, 5), on_center=False).construct(3)
conv = Convolution()
image = dataset[sample_idx][0].squeeze(0)
print(image.shape)
img = conv.execute(image, filter, padding=False)

norm_img = img.clone()
norm_img[norm_img < 0] = 0
enc = Intensity2Latency(time=15, dt=1.0)
import numpy as np

plt.style.use('seaborn-white')
plt_idx = 0
sw = enc(norm_img.unsqueeze(0))
print(sw.shape)
for t in range(6):
    plt_idx += 1
    ax = plt.subplot(5, 6, plt_idx)
    plt.setp(ax, xticklabels=[])
    plt.setp(ax, yticklabels=[])
    if t == 0:
        ax.set_ylabel('Feature ' + str(0))
    plt.imshow(sw[t].numpy(), cmap='gray')
plt.show()

# exit()
#
# exit()
time = 25  # time of simulation in ms
dt = 1.0  # dt in ms
neuron = LIFPopulation(shape=(5, 5, 5), v_th=-50.0, v_rest=-70.0, v_reset=-75.0, tau_m=10, r_m=10, t_ref=0,
                       init_t_ref=0, dt=dt, additive_spike_trace=True, tau_s=100)
input = torch.zeros((25, 5, 5, 5))
input[10:15, 0, 0, 0] = 0.0
input[10:15, 3, 2, 3] = 0.0
input[10:15, 4, 2, 3] = 7500.0
input[10:15, 4, 0, 0] = 7500.0

wtka = WinnerTakeAllDecision(kwta=2, inhibition_radius=1)
monitor = Monitor(neuron, state_variables=["s", "v", "current"])
monitor.reset_state_variables()
for t in range(0, int(time / dt)):
    neuron.forward(current=input[t])
    res, winners, inhibition_pot = wtka.compute(neuron)
    neuron.inject_v = inhibition_pot
    print(res, winners)
    monitor.record()
s = monitor.get("s")
v = monitor.get("v")
current = monitor.get("current")
