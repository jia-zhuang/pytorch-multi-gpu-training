import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"  # 必须在`import torch`语句之前设置才能生效
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Net
from data import train_dataset

device = torch.device('cuda')
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Net()
model = model.to(device)  # 默认会使用第一个GPU
optimizer = optim.SGD(model.parameters(), lr=0.1)

# training!
tb_writer = SummaryWriter(comment='single-gpu-training')
for i, (inputs, labels) in enumerate(train_loader):
    # forward
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs, labels=labels)
    loss = outputs[0]  # 对应模型定义中，模型返回始终是tuple
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # log
    if i % 10 == 0:
        tb_writer.add_scalar('loss', loss.item(), i)
tb_writer.close()