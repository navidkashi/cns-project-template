from cnsproject.preprocessing.filter import DoG, Gabor
from cnsproject.preprocessing.convolution import Convolution
import torch
import matplotlib.pyplot as plt
from cnsproject.encoding.encoders import *
import math
image_name = 'image2.png'
filter = DoG((1, 5), on_center=True).construct(31)
# filter = Gabor(6.0, 6.0, math.pi/2, 1, on_center=True).construct(10)
conv = Convolution()
image = plt.imread(image_name)
enc = PoissonEncoder(4, 1, max_rate=1000)
encoded = enc(torch.tensor(image, dtype=torch.float)).float()

img = conv.execute(encoded[1], filter, padding=True)
norm_img = img.clone()
norm_img[norm_img < 0] = 0
plt.imshow(norm_img, cmap='gray')
plt.show()
exit()
time = 50  # msec
dt = 1  # msec

norm_img = img.clone()
norm_img[norm_img < 0] = 0
norm_img = norm_img - norm_img.min()
norm_img = norm_img / norm_img.max()
norm_img = norm_img * 255

enc = Time2FirstSpikeEncoder(time, dt)
encoded = enc(torch.tensor(norm_img.tolist(), dtype=torch.float))

# enc = PoissonEncoder(time, dt, max_rate=100)
# encoded = enc(torch.tensor(norm_img.tolist(), dtype=torch.float))

plt.rcParams["figure.figsize"] = 20, 10
fig = plt.figure()
gs = fig.add_gridspec(2, 7, wspace=0.1, hspace=0.1)
axs0 = list()
axs1 = list()
for i in range(7):
    axs0.append(fig.add_subplot(gs[0, i]))
    axs0[i].set_yticks([])
    axs0[i].set_xticks([])
axs1.append(fig.add_subplot(gs[1, 0:3]))
axs1.append(fig.add_subplot(gs[1, 4:7]))
axs1[0].set_yticks([])
axs1[0].set_xticks([])
axs1[1].set_yticks([])
axs1[1].set_xticks([])
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
axs0[6].imshow(norm_img, cmap='gray')
axs0[6].set_title('Stimulus Image')
axs1[0].imshow(filter, cmap='gray')
axs1[0].set_title('Filter')
axs1[1].imshow(img, cmap='gray')
axs1[1].set_title('Filtered Image')
fig.show()
fig.savefig('first_dog_on_10_1_5_' + image_name + '.png', bbox_inches='tight')

