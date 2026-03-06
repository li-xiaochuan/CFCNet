import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, x):
        return self.main(x) + x


class LFEM(nn.Module):
    def __init__(self, dim, num_res=4):
        super(LFEM, self).__init__()
        layers = [ResBlock(dim) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class GFEM(nn.Module):
    def __init__(self, dim):
        super(GFEM, self).__init__()
        self.dim = dim
        self.norm = nn.BatchNorm2d(dim)
        self.conv_1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv_3 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')
        self.conv_5 = nn.Conv2d(dim, dim, kernel_size=3, padding=5, groups=dim, dilation=5, padding_mode='reflect')
        self.conv_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=7, groups=dim, dilation=7, padding_mode='reflect')
        self.conv_1x1 = nn.Conv2d(dim * 4, dim, kernel_size=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv_1(x)
        x1 = self.conv_3(x)
        x2 = self.conv_5(x1 + x)
        x3 = self.conv_7(x2 + x)
        x = torch.cat([x, x1, x2, x3], dim=1)
        x = self.conv_1x1(x)
        return x

class StandardFRM(nn.Module):
    def __init__(self, dim, num_res=1):
        super(StandardFRM, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 1)
        )
        self.local_branch = nn.Sequential(*[ResBlock(dim // 2) for _ in range(num_res)])
        self.global_branch = nn.Sequential(*[ResBlock(dim // 2) for _ in range(num_res)])


    def forward(self, x):
        identity = x
        x_local, x_global = x.chunk(2, dim=1)
        x_local = self.local_branch(x_local)
        x_global = self.global_branch(x_global)
        x = torch.cat([x_local, x_global], dim=1)
        x = identity + x
        x = self.mlp(x)
        return x

class FDEM(nn.Module):
    def __init__(self, dim, expand=2):
        super(FDEM, self).__init__()
        self.amplitude_mlp = nn.Sequential(
            nn.Conv2d(dim, expand * dim, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * dim, dim, 1, 1, 0)
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 1)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag_enhanced = self.amplitude_mlp(mag)
        real = mag_enhanced * torch.cos(pha)
        imag = mag_enhanced * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x_out = self.mlp(x_out)
        return x_out


def get_fft2freq(d1, d2, use_rfft=True):
    freq_h = torch.fft.fftfreq(d1)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)
    else:
        freq_w = torch.fft.fftfreq(d2)

    h_grid, w_grid = torch.meshgrid(freq_h, freq_w, indexing='ij')
    freq_grid = torch.stack([h_grid, w_grid], dim=-1)

    dist = torch.norm(freq_grid, dim=-1)

    _, indices = torch.sort(dist.reshape(-1))

    w_dim = d2 // 2 + 1 if use_rfft else d2
    sorted_coords = torch.stack([indices // w_dim, indices % w_dim], dim=-1)
    return sorted_coords.permute(1, 0)


class MDFM(nn.Module):
    def __init__(self, dim, kernel_size=3, kernel_num=4, group=8):
        super(MDFM, self).__init__()

        self.group = group
        self.dim = dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num

        self.conv1 = nn.Conv2d(dim, group, kernel_size=1, stride=1, groups=group, bias=False)
        self.conv2 = nn.Conv2d(group * kernel_num, group * kernel_num * kernel_size ** 2, kernel_size=1, stride=1, groups=group * kernel_num, bias=False)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.pad = nn.ReflectionPad2d(1)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm2d(group * kernel_num * 3 ** 2)
        self.act1 = nn.Softmax(dim=-2)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 4, dim, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim * 4, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1, stride=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim * 2, 1)
        )

    def forward(self, x_f):
        b, c, h, w = x_f.shape
        x_fft = self.conv1(x_f)

        freq_indices = get_fft2freq(h, w)
        total_freq = freq_indices.shape[1]
        x_fft = torch.fft.rfft2(x_fft, norm='ortho')
        group_size = total_freq // self.kernel_num
        groups = []
        for i in range(self.kernel_num):
            start = i * group_size
            end = (i+1)*group_size if i < self.kernel_num-1 else total_freq
            groups.append(freq_indices[:, start:end])
        masks = torch.zeros(self.kernel_num, h, w//2+1, device=x_f.device)
        for i, group in enumerate(groups):
            masks[i][group[0], group[1]] = 1.0
        masked_fft = x_fft.unsqueeze(2) * masks.unsqueeze(0).unsqueeze(0)
        x_recon = torch.fft.irfft2(masked_fft, s=(h, w), norm='ortho').reshape(b, self.group * self.kernel_num, h, w)

        dfilter = self.ap(x_recon)
        dfilter = self.conv2(dfilter)
        dfilter = self.bn(dfilter)

        x = F.unfold(self.pad(x_f), kernel_size=3).reshape(b, self.group, c // self.group, 9, h * w).unsqueeze(2)

        b, c1, p, q = dfilter.shape
        dfilter = dfilter.reshape(b, self.group, self.kernel_num, 1, 9, p * q)
        dfilter = self.act1(dfilter)
        x = torch.sum(x * dfilter, dim=4).reshape(b, c * self.kernel_num, h, w)

        x = x * self.sca(x)
        x = self.mlp(x)

        return x


class FRM(nn.Module):
    def __init__(self, dim, is_mdfm=False, is_fdem=False):
        super(FRM, self).__init__()
        self.dim = dim
        out_dim = dim * 2
        self.is_mdfm = is_mdfm
        self.is_fdem = is_fdem
        self.dw_conv = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, dilation=1, padding_mode='reflect')
        self.norm = nn.BatchNorm2d(out_dim)
        self.fusion = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=1),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
                                    nn.BatchNorm2d(dim),
                                    nn.GELU(),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
                                    nn.Conv2d(dim, dim, kernel_size=1))

        if is_mdfm:
            self.dyf = MDFM(dim)
        elif is_fdem:
            self.dyf = FDEM(dim)
        else:
            self.dyf = StandardFRM(dim)

        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, kernel_size=1),
        ) if is_mdfm else nn.Identity()

        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x_local, x_global, x_global_init):
        x_fused = torch.cat([x_local, x_global], dim=1)

        identity = x_fused
        x_fused = identity + self.dw_conv(x_fused)

        identity = x_fused
        x_fused = self.norm(x_fused)
        x_fused = self.fusion(x_fused)
        x_fused = self.dyf(x_fused)

        x_fused = identity + x_fused

        if self.is_mdfm:
            x = self.mlp(x_fused)
            return x
        else:
            x_local_new, x_global_new = torch.chunk(x_fused, 2, dim=1)
            x_global_new = x_global_new * self.alpha + x_global_init * self.beta
            return x_local_new, x_global_new

