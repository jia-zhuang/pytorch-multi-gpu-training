from torchvision import datasets, transforms

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
