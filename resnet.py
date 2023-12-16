import os
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from torchvision import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.to(device)
    return model


def preprocess_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def extract_features(model, image_tensor):
    with torch.no_grad():
        model.eval()
        features = model(image_tensor.to(device))
        features = features.squeeze()
        model.train()
        return features


def compute_distances(query_features, dataset_features):
    query_features_cpu = query_features.cpu().numpy()
    dataset_features_cpu = dataset_features.cpu().numpy()
    distances = [cosine(query_features_cpu, features) for features in dataset_features_cpu]
    return distances


def get_closest_images(query_image_path, model):
    query_tensor = preprocess_image(query_image_path)

    query_features = extract_features(model, query_tensor)

    dataset_features = []
    for batch_images in data_loader:
        batch_features = extract_features(model, batch_images[0])
        dataset_features.append(batch_features)

    dataset_features = torch.cat(dataset_features, dim=0)

    distances = compute_distances(query_features.squeeze(), dataset_features.squeeze())

    closest_indices = np.argsort(distances)[:6]

    return closest_indices


if __name__ == "__main__":
    query_image_path = "jpg/image_01981.jpg"

    resnet50_model = load_resnet50()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root="flowers", transform=transform)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False)

    closest_images = get_closest_images(query_image_path, resnet50_model)

    fig, axes = plt.subplots(2, 3, figsize=(8, 8))

    jpg_files = [f for f in os.listdir("jpg") if f.endswith('.jpg')]

    axes[0, 0].imshow(plt.imread(f"jpg/{jpg_files[closest_images[0]]}"))
    axes[0, 1].imshow(plt.imread(f"jpg/{jpg_files[closest_images[1]]}"))
    axes[0, 2].imshow(plt.imread(f"jpg/{jpg_files[closest_images[2]]}"))
    axes[1, 0].imshow(plt.imread(f"jpg/{jpg_files[closest_images[3]]}"))
    axes[1, 1].imshow(plt.imread(f"jpg/{jpg_files[closest_images[4]]}"))
    axes[1, 2].imshow(plt.imread(f"jpg/{jpg_files[closest_images[5]]}"))

    for ax in axes.flatten():
        ax.axis('off')

    plt.show()





