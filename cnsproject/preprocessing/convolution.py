from cnsproject.preprocessing.filter import *
import torch
import torch.nn.functional as F

class Convolution:
    def execute(self, image, kernel, padding=False, stride=1):
        if padding:
            pad_size = (kernel.shape[0] // 2, kernel.shape[1] // 2)
            padded_image = torch.zeros((image.shape[0] + pad_size[0] * 2, image.shape[1] + pad_size[1] * 2))
            padded_image[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]] = image
        else:
            padded_image = image.view(*image.shape)
        zero_mean_kernel = kernel - kernel.mean()
        output_shape = ((padded_image.shape[0] - kernel.shape[0]) // stride + 1, (padded_image.shape[1] - kernel.shape[1]) // stride + 1)
        output = torch.zeros(output_shape)
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                output[i, j] = torch.sum(torch.multiply(padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]], zero_mean_kernel))
        return output

class TorchConvolution:
    def execute(self, image, kernel, padding=False, stride=1):
        if padding:
            pad_size = (kernel.shape[0] // 2, kernel.shape[1] // 2)
        else:
            pad_size = (0, 0)
        zero_mean_kernel = kernel - kernel.mean()
        output = F.conv2d(
            image.unsqueeze(0).unsqueeze(0),
            zero_mean_kernel.unsqueeze(0).unsqueeze(0),
            stride=(stride, stride),
            padding=pad_size,
        ).squeeze()
        # print(output.shape)
        return output
