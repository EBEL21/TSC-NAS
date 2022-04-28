import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math


# https://github.com/AberHu/TF-NAS/blob/master/models/model_search.py
# https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py

def calculate_activation_size(x: torch.Tensor, m: nn.Module):
    activation = 0
    if isinstance(m, nn.Conv1d):
        activation = 0


# TODO: SE block추가
class SEBlock1d(nn.Module):
    def __init__(self, in_channels, r):
        super(SEBlock1d, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.ModuleList(
            [nn.Linear(in_channels, in_channels // r, bias=False),
             nn.ReLU(inplace=True),
             nn.Linear(in_channels // r, in_channels, bias=False),
             nn.Sigmoid()]
        )

    def forward(self, x):
        bs, c, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        # y = self.excitation(y).view(bs, c, 1)
        for m in self.excitation:
            y = m(y)
        y = y.view(bs, c, 1)
        return x * y.expand_as(x)


class NASConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kss, activation, device):
        super(NASConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.channel_ratio = [0.25, 0.5, 0.75, 1.0]
        self.channel_candidate = []
        self.channel_mask = self.create_channel_mask().to(device)

        self.se_factor = 8
        self.se_block = SEBlock1d(out_channels, self.se_factor)
        self.se_switch = 0

        self.kss = kss
        self.conv_layer = nn.ModuleList(
            [nn.Conv1d(in_channels, out_channels, kernel_size=k, bias=False, padding='same') for k in kss]
        )
        self.kernel_arch = torch.zeros(len(kss))

        self.bn = nn.BatchNorm1d(out_channels)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'relu6':
            self.act = nn.ReLU6()
        elif activation == 'hswish':
            self.act = nn.Hardswish()

        self.init_arch_params()
        self.reset_switches()

        self.bn_param_size = 0
        self.se_param_size = [0, 0]
        self.conv_param_size = []

    def init_arch_params(self):

        k_alphas = F.log_softmax(self.kernel_arch, dim=-1)
        self.register_parameter('k_alphas', nn.Parameter(k_alphas))

        channel_arch = torch.zeros((len(self.channel_ratio),))
        self.register_parameter('ch_alphas', nn.Parameter(channel_arch))

        se_alphas = torch.zeros(2)
        self.register_parameter('se_alphas', nn.Parameter(se_alphas))

    def set_temperature(self, T):
        self.T = T

    def create_channel_mask(self):
        channel_masks = []
        for ratio in self.channel_ratio:
            num_1s = int(self.out_channels * ratio)
            self.channel_candidate.append(num_1s)
            ones = torch.ones(num_1s)
            zeros = torch.zeros(self.out_channels - num_1s)
            mask = torch.concat((ones, zeros))
            channel_masks.append(mask)
        return torch.stack(channel_masks)

    def reset_switches(self):
        self.switch = [True] * len(self.kss)

    def fink_ori_idx(self, idx):
        count = 0
        for ori_idx in range(len(self.switch)):
            if self.switch[ori_idx]:
                count += 1
                if count == (idx + 1):
                    break
        return ori_idx

    def forward(self, x, sampling, mode, prev_ch=None):

        data_len = x.shape[2]
        ch_weights = F.softmax(self.ch_alphas, dim=-1)
        expected_channels = sum((cc * cw for cc, cw in zip(self.channel_candidate, ch_weights)))
        ch_mask = ch_weights.unsqueeze(1) * self.channel_mask
        ch_mask = torch.sum(ch_mask, dim=0)

        model_size = 0
        act_size = 0
        se_model_size = 0
        se_act_size = 0

        if sampling:
            weights = self.k_alphas[self.switch]
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

            x = self.conv_layer[idx](x)
            if self.se_switch == 1:
                x = self.se_block(x)
            self.se_switch = (self.se_switch + 1) % 2
        else:
            k_weights = F.gumbel_softmax(self.k_alphas, self.T, hard=False)
            x = sum(conv(x) for conv, k in zip(self.conv_layer, k_weights))

            se_out = self.se_block(x)

            se_weights = F.softmax(self.se_alphas, dim=-1)
            x = x * se_weights[0] + se_out * se_weights[1]

            # calculate model size
            # 1DConv -> se -> bn -> relu
            if prev_ch is not None:
                # conv_layer
                # param size (kernel_size x input_channel x output_channel)
                k_candidate = sum((k * kw for k, kw in zip(self.kss, k_weights)))
                model_size += prev_ch * expected_channels * k_candidate
                # activation size (batch_size x output_channel x length_out)
                act_size += expected_channels * data_len * 4

                # se block size
                # GAP -> Linear -> relu -> Linear -> Sigmoid
                se_model_size += expected_channels * (expected_channels // self.se_factor) * 2
                # activation_size (batch_size x expected_channels), (batch_size x expected_channels // se_factor)
                se_act_size += expected_channels // self.se_factor * 4  # Linear
                se_act_size += expected_channels // self.se_factor / 8  # ReLU
                se_act_size += expected_channels * 4  # Linear
                se_act_size += expected_channels * 4  # Sigmoid

                model_size += sum(sw * ms for sw, ms in zip(se_weights, (0, se_model_size)))
                act_size += sum(sw * act for sw, act in zip(se_weights, (0, se_act_size)))

                # bn size -> just channel size
                model_size += expected_channels * 2
                # activation_size
                act_size += expected_channels * data_len * 4

                # ReLU size
                # activation_size
                act_size += expected_channels * data_len / 8

                # print(model_size, act_size)

        x = x * ch_mask.unsqueeze(1)
        x = self.bn(x)
        output = self.act(x)

        memory_footprint = model_size + act_size

        return output, expected_channels, memory_footprint


# TODO: RNN based NAS block 정의
class NASRNN(nn.Module):
    def __init__(self, in_channels, hid_channels, channel_ratio=[0.6, 0.8, 1.0], device='cuda'):
        super(NASRNN, self).__init__()

        # self.module = nn.LSTM(in_channels, hid_channels, num_layers=1, bias=False, batch_first=True, dropout=0.5)

        self.in_channels = in_channels
        self.out_channels = hid_channels

        self.channel_ratio = [0.2, 0.6, 1.0]
        self.channel_candidate = []
        self.channel_arch = torch.zeros((len(self.channel_ratio),))
        self.channel_mask = self.create_channel_mask().to(device)

        self.device = device

        self.in2hid = nn.Linear(in_channels, 4 * hid_channels, bias=False)
        self.hid2hid = nn.Linear(hid_channels, 4 * hid_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.init_arch_params()

    def init_arch_params(self):
        channel_alphas = F.log_softmax(self.channel_arch, dim=-1)
        self.register_parameter('ch_alphas', nn.Parameter(channel_alphas))

    def create_channel_mask(self):
        channel_masks = []
        for ratio in self.channel_ratio:
            num_1s = int(self.out_channels * ratio)
            self.channel_candidate.append(num_1s)
            ones = torch.ones(num_1s)
            zeros = torch.zeros(self.out_channels - num_1s)
            mask = torch.concat((ones, zeros))
            channel_masks.append(mask)
        return torch.stack(channel_masks)

    def forward(self, x, sampling, mode):
        x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        h0 = torch.zeros(batch_size, self.out_channels).to(self.device)
        c0 = torch.zeros(batch_size, self.out_channels).to(self.device)

        weights = F.softmax(self.ch_alphas, dim=-1)
        expected_channels = sum((cc * cw for cc, cw in zip(self.channel_candidate, weights)))
        weights = weights.unsqueeze(1) * self.channel_mask
        weights = torch.sum(weights, dim=0)

        out = []
        h_t = h0
        c_t = c0
        for seq in range(x.size(1)):
            _input = x[:, seq, :]
            i_tmp = self.in2hid(_input)
            h_tmp = self.hid2hid(h_t)
            gates = i_tmp + h_tmp
            i_g, f_g, c_g, o_g = gates.squeeze(0).chunk(4, 1)

            i_g = i_g * weights
            f_g = f_g * weights
            c_g = c_g * weights
            o_g = o_g * weights

            i_g = self.sigmoid(i_g)
            f_g = self.sigmoid(f_g)
            o_g = self.sigmoid(o_g)

            c_t = f_g * c_t + i_g * self.tanh(c_g)
            h_t = o_g * self.tanh(c_t)
            out.append(h_t)

        model_size = 0
        act_size = 0
        # calculate model size
        # (batch_size x channel_size x 4) x 2
        model_size += expected_channels * 2 * 4

        # act size (batch_size x channel_size) x data_length
        # gate: 4x2
        # sigmoid: 3
        # tanh : 2
        act_size += x.size(1) * expected_channels * 13 * 4

        out = torch.stack(out)
        return h_t, expected_channels, model_size + act_size
