from model_util import get_models_dict
from torchvision import datasets, transforms
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import testers
import umap
import logging

import matplotlib.pyplot as plt
import numpy as np
import umap
from cycler import cycler
import faiss

test_transformation = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    logging.info(
        "UMAP plot for the {} split and label set {}".format(split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    plt.show()
    
if __name__ == '__main__':
    models = get_models_dict()
    trunk = models['trunk']
    embedder = models['embedder']
    
    test_dataset = datasets.Flowers102(root='.', split='test', download=True, transform=test_transformation)
        
    tester = testers.GlobalEmbeddingSpaceTester(
        dataloader_num_workers=1,
        accuracy_calculator=AccuracyCalculator(k=10),
        visualizer=umap.UMAP(),
        visualizer_hook=visualizer_hook,
    )
    
    accuracies = tester.test({'test': test_dataset}, epoch=0, trunk_model=trunk, embedder_model=embedder)
    print(accuracies)