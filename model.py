import torch.nn as nn
from layers import *


# TODO beta parameter 적용하기
class _RNN_FCN_Base(nn.Module):
    def __init__(self, c_in, c_out, num_blocks, activation='relu',
                 conv_max_channels=[128, 256, 128], kss=[3, 5, 7], min_depth=3, lstm_channels=256, device='cuda'):
        super(_RNN_FCN_Base, self).__init__()
        # FCN
        # assert len(conv_max_channels) == len(kss)
        self.c_in = c_in
        self.c_out = c_out

        self.device = device
        self.num_blocks = num_blocks
        self.min_depth = min_depth
        conv_max_channels.insert(0, c_in)

        self.conv_layers = nn.ModuleList(
            [NASConv1d(conv_max_channels[i], conv_max_channels[i + 1], kss, activation, device) for i in range(num_blocks)]
        )

        # RNN
        self.rnn = NASRNN(c_in, 256)

        # print(self.conv_layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv_max_channels[-1] + lstm_channels, c_out)
        self.initialize_betas()

    def weight_params(self):
        _weight_params = []

        for k, v in self.named_parameters():
            if not (k.endswith('alphas') or k.endswith('betas')):
                _weight_params.append(v)
        return _weight_params

    def arch_params(self):
        _arch_params = []

        for k, v in self.named_parameters():
            if k.endswith('alphas') or k.endswith('betas'):
                _arch_params.append(v)
        return _arch_params

    def initialize_betas(self):
        betas = torch.zeros((self.num_blocks - self.min_depth + 1))
        self.register_parameter('betas', nn.Parameter(betas))

    def set_temperature(self, T):
        for m in self.modules():
            if isinstance(m, NASConv1d):
                m.set_temperature(T)

    def reset_switches(self):
        for m in self.modules():
            if isinstance(m, NASConv1d):
                m.reset_switches()

    def forward(self, x, sampling, mode='gumbel'):
        batch_size = 8
        out_list = []
        mem_list = []
        ch_list = []
        mem_total = 0

        # LSTM
        lstm_out, ch_lstm, mem_lstm = self.rnn(x, sampling, mode)
        mem_total += mem_lstm
        # FCN
        ch_candidate = 1
        for idx, conv in enumerate(self.conv_layers):
            x, ch_candidate, memory = conv(x, sampling, mode, prev_ch=ch_candidate)

            if idx >= self.min_depth:
                out_list.append(x)
                mem_list.append(memory)
                ch_list.append(ch_candidate)
            else:
                mem_total += memory

        weights = F.softmax(self.betas, dim=0)
        conv_out = sum(w * res for w, res in zip(weights, out_list))
        conv_out = self.gap(conv_out)
        conv_out = conv_out.view(x.shape[0], -1)

        memory_out = sum(w * mem for w, mem in zip(weights, mem_list))
        mem_total += memory_out

        last_channel_candidate = sum(w * ch for w, ch in zip(weights, ch_list))

        x = torch.cat((conv_out, lstm_out), dim=1)
        x = self.fc(x)
        # print(x.shape)

        # fc memory
        # (lstm_channel + last_conv_channel) * fc_out
        mem_total += (last_channel_candidate + ch_lstm) * self.c_out * 4
        mem_total += self.c_out * 4
        mem_total = mem_total * batch_size / 1048576

        return x, mem_total
