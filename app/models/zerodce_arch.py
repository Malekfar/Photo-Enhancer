"""Zero-DCE++ architecture.

Vendored from the official repository https://github.com/Li-Chongyi/Zero-DCE_extension
(Zero-DCE++/model.py). Module and parameter names are unchanged so the official
Epoch99.pth snapshot loads with strict=True.

`curve_params` / `apply_curves` split the original forward() so the (low-frequency)
curve-parameter map can be estimated on a downscaled copy and the per-pixel curve applied
at full resolution, which is exactly what the original scale_factor mechanism does.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))


class enhance_net_nopool(nn.Module):
    curve_iterations = 8

    def __init__(self, scale_factor=1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32
        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv7 = CSDN_Tem(number_f * 2, 3)

    @staticmethod
    def apply_curves(x, x_r, iterations=8):
        for _ in range(iterations):
            x = x + x_r * (torch.pow(x, 2) - x)
        return x

    def curve_params(self, x_down):
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        return torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")
        x_r = self.curve_params(x_down)
        if self.scale_factor != 1:
            x_r = self.upsample(x_r)
        return self.apply_curves(x, x_r, self.curve_iterations), x_r
