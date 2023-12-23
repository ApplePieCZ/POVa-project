import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder

from model_util import get_models_dict

img_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

def print_decision(is_match):
    if is_match:
        print("Same class")
    else:
        print("Different class")


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
)
img_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


def imshow(img, figsize=(8, 4)):
    img = inv_normalize(img)
    npimg = img.numpy()
    plt.figure(figsize=figsize)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
if __name__ == '__main__':
    dataset = datasets.Flowers102(root=".", split='test', transform=img_transform, download=True)
    labels_to_indices = c_f.get_labels_to_indices(dataset._labels)
    
    models = get_models_dict()    
    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.7)
    inference_model = InferenceModel(trunk=models['trunk'], embedder=models['embedder'], match_finder=match_finder)

    inference_model.train_knn(dataset)

    classA, classB = labels_to_indices[3], labels_to_indices[4]
    
    for img_type in [classA, classB]:
        img = dataset[img_type[0]][0].unsqueeze(0)
        print("query image")
        imshow(torchvision.utils.make_grid(img))
        distances, indices = inference_model.get_nearest_neighbors(img, k=10)
        nearest_imgs = [dataset[i][0] for i in indices.cpu()[0]]
        print("nearest images")
        imshow(torchvision.utils.make_grid(nearest_imgs))    
    
    print("done model loading")
