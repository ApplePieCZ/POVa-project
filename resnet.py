# POVa Project - CBIR
# Lukas Marek
# 20.12.2023
import os
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from torchvision import datasets
import time
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_resnet50():
    # Load the pretrained model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Pretty good
    # model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)  # Good
    # model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)  # Good
    model.to(device)
    return model


def preprocess_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def extract_features(model, image_tensor):
    # Extract features from image on GPU using ResNet
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


def get_closest_images(image_path, model):
    # Preprocess the query image
    query_tensor = preprocess_image(image_path)

    # Extract features from the query image
    query_features = extract_features(model, query_tensor)

    dataset_features = []
    for batch_images in data_loader:
        batch_features = extract_features(model, batch_images[0])
        dataset_features.append(batch_features)

    # Concatenate features from all batches
    dataset_features = torch.cat(dataset_features, dim=0)

    # Compute distances
    distances = compute_distances(query_features.squeeze(), dataset_features.squeeze())

    # Get indices of 6 closest images
    closest_indices = np.argsort(distances)[:6]

    return closest_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resnet based CBIR.")
    parser.add_argument("-i", type=str, help="Query image", required=True)
    parser.add_argument("-d", type=str, help="Path to reference images")
    parser.add_argument("-r", type=str, help="Path to dataset root")
    parser.add_argument("-c", type=int, help="Size of dataset to clip")
    parser.add_argument("-f", action='store_true', help="If features are stored.")

    args = parser.parse_args()

    query_image_path = args.i

    reference_images_path = args.d

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    resnet50_model = load_resnet50()

    start_time = time.time()

    if args.f:
        query_tensor = preprocess_image(query_image_path)
        query_features = extract_features(resnet50_model, query_tensor)
        loaded_features_array = np.load('features.npy')
        loaded_features_tensor = torch.from_numpy(loaded_features_array)[:args.c]
        distances = compute_distances(query_features.squeeze(), loaded_features_tensor.squeeze())
        closest_images = np.argsort(distances)[:6]
    else:
        dataset = datasets.ImageFolder(root=args.r, transform=transform)

        if args.c < len(dataset):
            subset_size = int(args.c)
        else:
            subset_size = len(dataset)

        subset_flowers = Subset(dataset, range(subset_size))

        data_loader = DataLoader(subset_flowers, batch_size=512, shuffle=False)

        closest_images = get_closest_images(query_image_path, resnet50_model)

    print(f"{time.time() - start_time:.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(8, 8))

    jpg_files = [f for f in os.listdir(reference_images_path) if f.endswith('.jpg')]

    axes[0, 0].imshow(plt.imread(f"{reference_images_path}/{jpg_files[closest_images[0]]}"))
    axes[0, 1].imshow(plt.imread(f"{reference_images_path}/{jpg_files[closest_images[1]]}"))
    axes[0, 2].imshow(plt.imread(f"{reference_images_path}/{jpg_files[closest_images[2]]}"))
    axes[1, 0].imshow(plt.imread(f"{reference_images_path}/{jpg_files[closest_images[3]]}"))
    axes[1, 1].imshow(plt.imread(f"{reference_images_path}/{jpg_files[closest_images[4]]}"))
    axes[1, 2].imshow(plt.imread(f"{reference_images_path}/{jpg_files[closest_images[5]]}"))

    for ax in axes.flatten():
        ax.axis('off')

    #plt.show()
