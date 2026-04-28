import numpy as np
import scipy.stats as stats
import torch
from torch import nn
from torch.nn import functional as F


def gabor_kernel(frequency, sigma_x, sigma_y, theta=0, offset=0, ks=61):
    half = ks // 2
    grid = torch.arange(-half, half + 1, dtype=torch.float)
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    rot_x = x * np.cos(theta) + y * np.sin(theta)
    rot_y = -x * np.sin(theta) + y * np.cos(theta)
    kernel = torch.exp(-0.5 * (rot_x ** 2 / sigma_x ** 2 + rot_y ** 2 / sigma_y ** 2))
    kernel /= 2 * np.pi * sigma_x * sigma_y
    kernel *= torch.cos(2 * np.pi * frequency * rot_x + offset)
    return kernel


def sample_dist(hist, bins, num_samples, scale="linear"):
    samples = np.random.rand(num_samples)
    if scale == "linear":
        return np.interp(samples, np.hstack(([0], hist.cumsum())), bins)
    if scale == "log2":
        return 2 ** np.interp(samples, np.hstack(([0], hist.cumsum())), np.log2(bins))
    if scale == "log10":
        return 10 ** np.interp(samples, np.hstack(([0], hist.cumsum())), np.log10(bins))
    raise ValueError(f"Unsupported sampling scale: {scale}")


def generate_gabor_param(features, seed=0, rand_flag=False, sf_corr=0.75, sf_max=9, sf_min=0):
    np.random.seed(seed)

    phase_bins = np.array([0, 360])
    phase_dist = np.array([1])

    if rand_flag:
        ori_bins = np.array([0, 180])
        ori_dist = np.array([1])
        nx_bins = np.array([0.1, 10 ** 0.2])
        nx_dist = np.array([1])
        ny_bins = np.array([0.1, 10 ** 0.2])
        ny_dist = np.array([1])
        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8])
        sf_dist = np.array([1, 1, 1, 1, 1, 1, 1, 1])

        sfmax_ind = np.where(sf_bins < sf_max)[0][-1]
        sfmin_ind = np.where(sf_bins >= sf_min)[0][0]
        sf_bins = sf_bins[sfmin_ind:sfmax_ind + 1]
        sf_dist = sf_dist[sfmin_ind:sfmax_ind]
        sf_dist = sf_dist / sf_dist.sum()
    else:
        ori_bins = np.array([-22.5, 22.5, 67.5, 112.5, 157.5])
        ori_dist = np.array([66, 49, 77, 54], dtype=np.float64)
        ori_dist = ori_dist / ori_dist.sum()

        cov_mat = np.array([[1, sf_corr], [sf_corr, 1]])
        nx_bins = np.logspace(-1, 0.2, 6, base=10)
        ny_bins = np.logspace(-1, 0.2, 6, base=10)
        n_joint_dist = np.array([
            [2.0, 0.0, 1.0, 0.0, 0.0],
            [8.0, 9.0, 4.0, 1.0, 0.0],
            [1.0, 2.0, 19.0, 17.0, 3.0],
            [0.0, 0.0, 1.0, 7.0, 4.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        n_joint_dist = n_joint_dist / n_joint_dist.sum()
        nx_dist = n_joint_dist.sum(axis=1)
        nx_dist = nx_dist / nx_dist.sum()
        ny_row_sums = n_joint_dist.sum(axis=1, keepdims=True)
        ny_dist_marg = np.divide(
            n_joint_dist,
            ny_row_sums,
            out=np.zeros_like(n_joint_dist),
            where=ny_row_sums != 0,
        )

        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8])
        sf_dist = np.array([4, 4, 8, 25, 32, 26, 28, 12], dtype=np.float64)

        sfmax_ind = np.where(sf_bins <= sf_max)[0][-1]
        sfmin_ind = np.where(sf_bins >= sf_min)[0][0]
        sf_bins = sf_bins[sfmin_ind:sfmax_ind + 1]
        sf_dist = sf_dist[sfmin_ind:sfmax_ind]
        sf_dist = sf_dist / sf_dist.sum()

    phase = sample_dist(phase_dist, phase_bins, features)
    ori = sample_dist(ori_dist, ori_bins, features)
    ori[ori < 0] = ori[ori < 0] + 180

    if rand_flag:
        sf = sample_dist(sf_dist, sf_bins, features, scale="log2")
        nx = sample_dist(nx_dist, nx_bins, features, scale="log10")
        ny = sample_dist(ny_dist, ny_bins, features, scale="log10")
    else:
        samps = np.random.multivariate_normal([0, 0], cov_mat, features)
        samps_cdf = stats.norm.cdf(samps)

        nx = np.interp(samps_cdf[:, 0], np.hstack(([0], nx_dist.cumsum())), np.log10(nx_bins))
        nx = 10 ** nx

        ny_samp = np.random.rand(features)
        ny = np.zeros(features)
        for sample_idx, nx_sample in enumerate(nx):
            bin_id = np.argwhere(nx_bins < nx_sample)[-1]
            ny[sample_idx] = np.interp(
                ny_samp[sample_idx],
                np.hstack(([0], ny_dist_marg[bin_id, :].cumsum())),
                np.log10(ny_bins),
            )
        ny = 10 ** ny

        sf = np.interp(samps_cdf[:, 1], np.hstack(([0], sf_dist.cumsum())), np.log2(sf_bins))
        sf = 2 ** sf

    return sf, ori, phase, nx, ny


class Identity(nn.Module):
    def forward(self, x):
        return x


class GFB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (kernel_size // 2, kernel_size // 2)
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

    def initialize(self, sf, theta, sigx, sigy, phase):
        random_channel = torch.randint(0, self.in_channels, (self.out_channels,))
        for idx in range(self.out_channels):
            self.weight[idx, random_channel[idx]] = gabor_kernel(
                frequency=sf[idx],
                sigma_x=sigx[idx],
                sigma_y=sigy[idx],
                theta=theta[idx],
                offset=phase[idx],
                ks=self.kernel_size[0],
            )
        self.weight = nn.Parameter(self.weight, requires_grad=False)


class VOneBlock(nn.Module):
    def __init__(
        self,
        sf,
        theta,
        sigx,
        sigy,
        phase,
        k_exc=25,
        noise_mode=None,
        noise_scale=1,
        noise_level=1,
        simple_channels=128,
        complex_channels=128,
        ksize=25,
        stride=4,
        input_size=224,
    ):
        super().__init__()
        self.in_channels = 3
        self.simple_channels = simple_channels
        self.complex_channels = complex_channels
        self.out_channels = simple_channels + complex_channels
        self.stride = stride
        self.input_size = input_size
        self.k_exc = k_exc
        self.fixed_noise = None

        self.set_noise_mode(noise_mode, noise_scale, noise_level)

        self.simple_conv_q0 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q1 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q0.initialize(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase)
        self.simple_conv_q1.initialize(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase + np.pi / 2)

        self.simple = nn.ReLU(inplace=True)
        self.complex = Identity()
        self.gabors = Identity()
        self.noise = nn.ReLU(inplace=True)
        self.output = Identity()

    def forward(self, x, add_noise=True):
        x = self.gabors_f(x)
        x = self.noise_f(x, add_noise=add_noise)
        return self.output(x)

    def gabors_f(self, x):
        s_q0 = self.simple_conv_q0(x)
        s_q1 = self.simple_conv_q1(x)
        complex_acts = self.complex(
            torch.sqrt(s_q0[:, self.simple_channels:, :, :] ** 2 + s_q1[:, self.simple_channels:, :, :] ** 2)
            / np.sqrt(2)
        )
        simple_acts = self.simple(s_q0[:, :self.simple_channels, :, :])
        return self.gabors(self.k_exc * torch.cat((simple_acts, complex_acts), dim=1))

    def noise_f(self, x, add_noise=True):
        if add_noise and self.noise_mode == "neuronal":
            eps = 1e-4
            x = x * self.noise_scale
            x = x + self.noise_level
            noise = self.fixed_noise
            if noise is None:
                noise = torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample()
            x = x + noise * torch.sqrt(F.relu(x.clone()) + eps)
            x = x - self.noise_level
            x = x / self.noise_scale
        elif add_noise and self.noise_mode == "gaussian":
            noise = self.fixed_noise
            if noise is None:
                noise = torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample()
            x = x + noise * self.noise_scale
        return self.noise(x)

    def set_noise_mode(self, noise_mode=None, noise_scale=1, noise_level=1):
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level

    def fix_noise(self, batch_size=256, seed=None):
        spatial_size = int(self.input_size / self.stride)
        noise_mean = torch.zeros(batch_size, self.out_channels, spatial_size, spatial_size)
        if seed is not None:
            torch.manual_seed(seed)
        if self.noise_mode:
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean, scale=1).rsample()

    def unfix_noise(self):
        self.fixed_noise = None


def build_v1_block(
    image_size=224,
    visual_degrees=8,
    stride=4,
    ksize=25,
    sf_corr=0.75,
    sf_max=9,
    sf_min=0,
    rand_param=False,
    gabor_seed=0,
    simple_channels=256,
    complex_channels=256,
    noise_mode="neuronal",
    noise_scale=0.35,
    noise_level=0.07,
    k_exc=25,
):
    out_channels = simple_channels + complex_channels
    sf, theta, phase, nx, ny = generate_gabor_param(
        out_channels,
        seed=gabor_seed,
        rand_flag=rand_param,
        sf_corr=sf_corr,
        sf_max=sf_max,
        sf_min=sf_min,
    )

    ppd = image_size / visual_degrees
    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta / 180 * np.pi
    phase = phase / 180 * np.pi

    block = VOneBlock(
        sf=sf,
        theta=theta,
        sigx=sigx,
        sigy=sigy,
        phase=phase,
        k_exc=k_exc,
        noise_mode=noise_mode,
        noise_scale=noise_scale,
        noise_level=noise_level,
        simple_channels=simple_channels,
        complex_channels=complex_channels,
        ksize=ksize,
        stride=stride,
        input_size=image_size,
    )
    block.gabor_params = {
        "simple_channels": simple_channels,
        "complex_channels": complex_channels,
        "rand_param": rand_param,
        "gabor_seed": gabor_seed,
        "sf_max": sf_max,
        "sf_corr": sf_corr,
    }
    return block
