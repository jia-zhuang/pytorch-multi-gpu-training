# PyTorch 单机多GPU 训练方法与原理整理

这里整理一些PyTorch单机多核训练的方法和简单原理，目的是既能在写代码时知道怎么用，又能从原理上知道大致是怎么回事儿。如果只是炼丹，有时候确实没时间和精力深挖太多实现原理，但又希望能理解简单逻辑。

PyTorch单机多核训练方案有两种：一种是利用`nn.DataParallel`实现，实现简单，不涉及多进程；另一种是用`torch.nn.parallel.DistributedDataParallel`和`torch.utils.data.distributed.DistributedSampler`结合多进程实现。第二种方式效率更高，但是实现起来稍难，第二种方式同时支持多节点分布式实现。方案二的效率要比方案一高，即使是在单运算节点上。

为方便理解，这里用一个简单的CNN模型训练MNIST手写数据集，相关代码：

- [model.py](./model.py)：定义一个简单的CNN网络
- [data.py](./data.py)：MNIST训练集和数据集准备
- [single_gpu_train.py](./single_gpu_train.py)：单GPU训练代码

### 方案一

核心在于使用`nn.DataParallel`将模型wrap一下，代码其他地方不需要做任何更改:

```python
model = nn.DataParallel(model)
```

为方便说明，我们假设模型输入为(32, input_dim)，这里的 32 表示batch_size，模型输出为(32, output_dim)，使用 4 个GPU训练。`nn.DataParallel`起到的作用是将这 32 个样本拆成 4 份，发送给 4 个GPU 分别做 forward，然后生成 4 个大小为(8, output_dim)的输出，然后再将这 4 个输出都收集到`cuda:0`上并合并成(32, output_dim)。

可以看出，`nn.DataParallel`没有改变模型的输入输出，因此其他部分的代码不需要做任何更改，非常方便。但弊端是，后续的loss计算只会在`cuda:0`上进行，没法并行，因此会导致负载不均衡的问题。

如果把`loss`放在模型里计算的话，则可以缓解上述负载不均衡的问题，示意代码如下：

```python

class Net:
    def __init__(self,...):
        # code
    
    def forward(self, inputs, labels=None)
        # outputs = fct(inputs)
        # loss_fct = ...
        if labels is not None:
            loss = loss_fct(outputs, labels)  # 在训练模型时直接将labels传入模型，在forward过程中计算loss
            return loss
        else:
            return outputs
```

按照我们上面提到的模型并行逻辑，在每个GPU上会计算出一个loss，这些loss会被收集到`cuda:0`上并合并成长度为 4 的张量。这个时候在做backward的之前，必须对将这个loss张量合并成一个标量，一般直接取mean就可以。这在Pytorch官方文档[nn.DataParallel函数]()中有提到：

> When `module` returns a scalar (i.e., 0-dimensional tensor) in forward(), this wrapper will return a vector of length equal to number of devices used in data parallelism, containing the result from each device.

这部分的例子可以参考：[data_parallel_train.py](./data_parallel.py)

### 方案二

方案二被成为分布式数据并行(distributed data parallel)，是通过多进程实现的，相比与方案一要复杂很多。可以从以下几个方面理解：

1. 从一开始就会启动多个进程(进程数等于GPU数)，每个进程独享一个GPU，每个进程都会独立地执行代码。这意味着每个进程都独立地初始化模型、训练，当然，在每次迭代过程中会通过进程间通信共享梯度，整合梯度，然后独立地更新参数。

2. 每个进程都会初始化一份训练数据集，当然它们会使用数据集中的不同记录做训练，这相当于同样的模型喂进去不同的数据做训练，也就是所谓的数据并行。这是通过`torch.utils.data.distributed.DistributedSampler`函数实现的，不过逻辑上也不难想到，只要做一下数据partition，不同进程拿到不同的parition就可以了，官方有一个简单的demo，感兴趣的可以看一下代码实现：[Distributed Training](https://pytorch.org/tutorials/intermediate/dist_tuto.html#distributed-training)

3. 进程通过`local_rank`变量来标识自己，`local_rank`为0的为master，其他是slave。这个变量是`torch.distributed`包帮我们创建的，使用方法如下：

    ```python
    import argparse  # 必须引入 argparse 包
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    ```

    必须以如下方式运行代码：

    ```bash
    python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py
    ```

    这样的话，`torch.distributed.launch`就以命令行参数的方式将`args.local_rank`变量注入到每个进程中，每个进程得到的变量值都不相同。比如使用 4 个GPU的话，则 4 个进程获得的`args.local_rank`值分别为0、1、2、3。

    上述命令行参数`nproc_per_node`表示每个节点需要创建多少个进程(使用几个GPU就创建几个)；`nnodes`表示使用几个节点，因为我们是做单机多核训练，所以设为1。

4. 因为每个进程都会初始化一份模型，为保证模型初始化过程中生成的随机权重相同，需要设置随机种子。方法如下：

    ```python
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    ```


使用方法通过如下示意代码展示：

```python
from torch.utils.data.distributed import DistributedSampler  # 负责分布式dataloader创建，也就是实现上面提到的partition。

# 负责创建 args.local_rank 变量，并接受 torch.distributed.launch 注入的值
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

# 每个进程根据自己的local_rank设置应该使用的GPU
torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)

# 初始化分布式环境，主要用来帮助进程间通信
torch.distributed.init_process_group(backend='nccl')

# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 初始化模型
model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 只 master 进程做 logging，否则输出会很乱
if args.local_rank == 0:
    tb_writer = SummaryWriter(comment='ddp-training')

# 分布式数据集
train_sampler = DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)  # 注意这里的batch_size是每个GPU上的batch_size

# 分布式模型
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
```

详细代码参考：[ddp_train.py](./ddp_train.py)


### ddp有用的技巧

官方推荐使用方案二(ddp)，所以这里收集ddp使用过程中的一些技巧。

#### torch.distributed.barrier

在读[huggingface/transformers](https://github.com/huggingface/transformers)中的源码，比如`examples/run_ner.py`会看到一下代码：

```python
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
```

上述代码要实现预训练模型的下载和读入内存，如果4个进程都分别下载一遍显然是不合理的，那如何才能实现只让一个进程下载呢？这个时候就可以使用`barrier`函数。当slave进程(local_rank!=0)运行到第一个`if`时就被barrier住了，只能等着，但master进程可以往下运行完成模型的下载和读入内存，但在第二个`if`语句时遇到barrier，那会不会被barrier住呢？答案是不会，因为master进程和slave进程集合在一起了(barrier)，barrier会被解除，这样大家都往下执行。当然这时大家执行的进度不同，master进程已经执行过模型读入，所以从第二个`if`往下执行，而slave进程尚未执行模型读入，只会从第一个`if`往下执行。

可以看到`barrier`类似一个路障，进程会被拦住，直到所有进程都集合齐了才放行。适合这样的场景：只一个进程下载，其他进程可以使用下载好的文件；只一个进程预处理数据，其他进程使用预处理且cache好的数据等。

#### 模型保存

模型的保存与加载，与单GPU的方式有所不同。这里通通将参数以cpu的方式save进存储, 因为如果是保存的GPU上参数，pth文件中会记录参数属于的GPU号，则加载时会加载到相应的GPU上，这样就会导致如果你GPU数目不够时会在加载模型时报错，像下面这样：
>RuntimeError: Attempting to deserialize object on CUDA device 1 but torch.cuda.device_count() is 1. Please use torch.load with map_location to map your storages to an existing device.

模型保存都是一致的，不过时刻记住方案二中你有多个进程在同时跑，所以会保存多个模型到存储上，如果使用共享存储就要注意文件名的问题，当然一般只在rank0进程上保存参数即可，因为所有进程的模型参数是同步的。

```python
torch.save(model.module.cpu().state_dict(), "model.pth")
```

模型的加载：

```python
param=torch.load("model.pth")
```

以下是[huggingface/transformers]()代码中用到的模型保存代码

```python
if torch.distributed.get_rank() == 0:
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
```

#### 同一台机器上跑多个 ddp task

假设想在一台有4核GPU的电脑上跑两个ddp task，每个task使用两个核，很可能会需要如下错误：

```
RuntimeError: Address already in use
RuntimeError: NCCL error in: /opt/conda/conda-bld/pytorch_1544081127912/work/torch/lib/c10d/ProcessGroupNCCL.cpp:260, unhandled system error
```

原因是两个ddp task通讯地址冲突，这时候需要显示地设置每个task的地址

> specifying a different master_addr and master_port in torch.distributed.launch

```bash
# 第一个task
export CUDA_VISIBLE_DEVICES="0,1" 
python -m torch.distributed.launch --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29501 train.py

# 第二个task
export CUDA_VISIBLE_DEVICES="2,3" 
python -m torch.distributed.launch --nproc_per_node=2 --master_addr=127.0.0.2 --master_port=29502 train.py
```


### 参考

[Pytorch 多GPU训练-单运算节点-All you need](https://www.cnblogs.com/walter-xh/p/11586507.html)

[WRITING DISTRIBUTED APPLICATIONS WITH PYTORCH](https://pytorch.org/tutorials/intermediate/dist_tuto.html)

[pytorch 多GPU训练总结（DataParallel的使用）](https://blog.csdn.net/weixin_40087578/article/details/87186613)
