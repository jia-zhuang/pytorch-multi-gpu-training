import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from model import Net
from data import train_dataset, test_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)
torch.distributed.init_process_group(backend='nccl')

batch_size = 64

model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# training!
if args.local_rank == 0:
    tb_writer = SummaryWriter(comment='ddp-3')

train_sampler = DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

for i, (inputs, labels) in enumerate(train_loader):
    # forward
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # log
    if args.local_rank == 0 and i % 5 == 0:
        tb_writer.add_scalar('loss', loss.item(), i)

if args.local_rank == 0:
    tb_writer.close()