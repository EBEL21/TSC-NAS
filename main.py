import torch.nn as nn
from torch.utils.data import DataLoader
from model import _RNN_FCN_Base
from load_data import TSCDataset
from train import *
import time
import logging, sys, os

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)

train_data = TSCDataset('ECG5000', var_type='Uni', split='train')
train_loader = DataLoader(train_data, batch_size=25, shuffle=True)

test_data = TSCDataset('ECG5000', var_type='Uni', split='test')
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

num_blocks = 5
activation = 'relu'
max_channels = [256] * num_blocks
kss = [3, 5, 7]
device = 'cuda'
from memory_cost_profiler import profile_memory_cost

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

T = 5.0
T_decay = 0.96

model = _RNN_FCN_Base(train_data.in_channel, train_data.out_channel, num_blocks,
                      activation, max_channels, kss, 3, 256, device=device)
model.to(device)

epochs = 200
w_lr = 3e-4
a_lr = 1e-2

optimizer_w = torch.optim.SGD(model.weight_params(), lr=w_lr)
optimizer_a = torch.optim.Adam(model.arch_params(), lr=a_lr)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

for epoch in range(epochs):
    # epoch_start = time.time()
    model.set_temperature(T)
    logging.info('Epoch: %d lr: %e T: %e', epoch, w_lr, T)
    if epoch < 10:
        train_acc = train_wo_arch(model, train_loader, optimizer_w, criterion)
    else:
        train_acc = train_w_arch(model, train_loader, test_loader, optimizer_w, optimizer_a, criterion)
        T *= T_decay

    # arch_params = model.arch_params()
    # for p in arch_params:
    #     print(p)
