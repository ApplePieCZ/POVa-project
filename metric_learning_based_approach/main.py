import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import umap
from cycler import cycler
from PIL import Image
from torchvision import datasets, transforms

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
        
class MLP(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)
    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.info("VERSION %s" % pytorch_metric_learning.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trunk = torchvision.models.resnet18(pretrained=True)
    trunk_output_size = trunk.fc.in_features
    trunk.fc = nn.Identity()
    trunk = torch.nn.DataParallel(trunk.to(device))

    embedder = torch.nn.DataParallel(MLP([trunk_output_size, 64]).to(device))

    # Set optimizers
    trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00001, weight_decay=0.0001)
    embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.0001, weight_decay=0.0001)

    train_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    train_dataset = datasets.Flowers102(root='.', split='train', download=True, transform=train_transform)
    val_dataset = datasets.Flowers102(root='.', split='val', download=True, transform=val_transform)
    
    loss = losses.TripletMarginLoss(margin=0.1)

    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    sampler = samplers.MPerClassSampler(train_dataset._labels, m=4, length_before_new_iter=len(train_dataset))

    batch_size = 32
    num_epochs = 4

    models = {"trunk": trunk, "embedder": embedder}
    optimizers = {
        "trunk_optimizer": trunk_optimizer,
        "embedder_optimizer": embedder_optimizer,
    }
    loss_funcs = {"metric_loss": loss}
    mining_funcs = {"tuple_miner": miner}
    
    record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", "example_tensorboard")
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"val": val_dataset}
    model_folder = "example_saved_models"

    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        dataloader_num_workers=2,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
    )

    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder, test_interval=1, patience=1)
    
    trainer = trainers.MetricLossOnly(
        models,
        optimizers,
        batch_size,
        loss_funcs,
        train_dataset,
        mining_funcs=mining_funcs,
        sampler=sampler,
        dataloader_num_workers=2,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )
    trainer.train(num_epochs=num_epochs)