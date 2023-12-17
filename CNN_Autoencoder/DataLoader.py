import torch
from torchvision import datasets


def get_DataLoader(split, transform, batch_size, shuffle):
    dataset = datasets.Flowers102(root='flowers',
                                  split=split,
                                  transform=transform,
                                  download=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle)
    return test_loader
