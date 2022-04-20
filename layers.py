import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# https://github.com/AberHu/TF-NAS/blob/master/models/model_search.py
# https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py

# TODO: SE block추가
class NASConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation):
        super(NASConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.channel_ratio = [0.25, 0.5, 0.75, 1.0]
        self.channel_arch = torch.zeros((len(self.channel_ratio),))
        self.channel_mask = self.create_channel_mask().to('cuda')

        self.se_ratio = [0.0, 1 / 16, 1 / 8, 1 / 4]
        self.se_arch = torch.zeros((len(self.se_ratio, )))

        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'relu6':
            self.act = nn.ReLU6()
        elif activation == 'hswish':
            self.act = nn.Hardswish()

        self.init_arch_params()
        self.reset_switches()

    def init_arch_params(self):
        channel_alphas = F.log_softmax(self.channel_arch, dim=-1)
        self.register_parameter('log_alphas', nn.Parameter(channel_alphas))

    def set_temperature(self, T):
        self.T = T

    def create_channel_mask(self):
        channel_masks = []
        for ratio in self.channel_ratio:
            num_1s = int(self.out_channels * ratio)
            ones = torch.ones(num_1s)
            zeros = torch.zeros(self.out_channels - num_1s)
            mask = torch.concat((ones, zeros))
            channel_masks.append(mask)
        return torch.stack(channel_masks)

    def reset_switches(self):
        self.switch = [True] * len(self.channel_ratio)

    def fink_ori_idx(self, idx):
        count = 0
        for ori_idx in range(len(self.switch)):
            if self.switch[ori_idx]:
                count += 1
                if count == (idx + 1):
                    break
        return ori_idx

    def forward(self, x, sampling, mode):
        if sampling:
            weights = self.log_alphas[self.switch]
            if mode == 'gumbel':
                weights = F.gumbel_softmax(F.log_softmax(weights, dim=-1), self.T, hard=False)
                idx = torch.argmax(weights).item()
                self.switch[idx] = False
            elif mode == 'random':
                idx = random.choice(range(len(weights)))
                idx = self.fink_ori_idx(idx)
                self.reset_switches()
            else:
                raise ValueError('invalid sampling mode')
            x = self.conv_layer(x)
            x = x * self.channel_mask[idx].unsqueeze(1)
        else:
            weights = F.gumbel_softmax(self.log_alphas, self.T, hard=False)
            weights = weights.unsqueeze(1) * self.channel_mask
            weights = torch.sum(weights, dim=0)
            x = self.conv_layer(x)
            x = x * weights.unsqueeze(1)
        x = self.act(x)
        output = self.bn(x)
        return output


# TODO: RNN based NAS block 정의
class NASRNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NASRNN, self).__init__()
        self.module = nn.LSTM