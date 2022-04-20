import torch.nn as nn
from layers import *

# TODO beta parameter 적용하기
class _RNN_FCN_Base(nn.Module):
    def __init__(self, c_in, c_out, num_blocks, activation='relu', max_channels=[128, 256, 128], kss=[7, 5, 3], se=0):
        super(_RNN_FCN_Base, self).__init__()
        # FCN
        assert len(max_channels) == len(kss)
        self.device = 'cuda'
        self.num_blocks = num_blocks
        max_channels.insert(0, c_in)

        self.conv_layers = nn.ModuleList(
            [NASConv1d(max_channels[i], max_channels[i+1], kss[i], activation) for i in range(num_blocks)]
        )

        # RNN
        self.rnn_layers = nn.ModuleList()

        # print(self.conv_layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(max_channels[-1], c_out)
        # self.initialize_betas()

    def weight_params(self):
        _weight_params = []

        for k, v in self.named_parameters():
            if not (k.endswith('log_alphas') or k.endswith('betas')):
                _weight_params.append(v)
        return _weight_params

    def arch_params(self):
        _arch_params = []

        for k, v in self.named_parameters():
            if k.endswith('log_alphas') or k.endswith('betas'):
                _arch_params.append(v)
        return _arch_params

    def initialize_betas(self):
        betas = torch.zeros(self.num_blocks)
        self.register_parameter('betas', nn.Parameter(betas))

    def set_temperature(self, T):
        for m in self.modules():
            if isinstance(m, NASConv1d):
                m.set_temperature(T)

    def forward(self, x, sampling, mode='gumbel'):

        # FCN
        for conv in self.conv_layers:
            x = conv(x, sampling, mode)
        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        # Concat
        # x = self.concat([last_out, x])
        # x = self.fc_dropout(x)
        x = self.fc(x)
        return x


class RNN_FCN(_RNN_FCN_Base):
    _cell = nn.RNN


class LSTM_FCN(_RNN_FCN_Base):
    _cell = nn.LSTM


class GRU_FCN(_RNN_FCN_Base):
    _cell = nn.GRU


class MRNN_FCN(_RNN_FCN_Base):
    _cell = nn.RNN

    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=se, **kwargs)


class MLSTM_FCN(_RNN_FCN_Base):
    _cell = nn.LSTM

    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=se, **kwargs)


class MGRU_FCN(_RNN_FCN_Base):
    _cell = nn.GRU

    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=se, **kwargs)

