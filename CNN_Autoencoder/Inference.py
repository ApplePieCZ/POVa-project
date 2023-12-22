import os
from datetime import datetime

import torch
from PIL import Image
from matplotlib import image as mpimg, pyplot as plt
from sklearn.neighbors import NearestNeighbors
import time
from torchvision import transforms as transforms

from LocalDataset import encode, load_dataset_paths, load_encoded_dataset

RESIZE = 32


def inference(autoencoder, only_process, image_src: list):
    autoencoder_model = load_autoencoder_model('conv_autoencoder.pth', autoencoder)
    en_dataset_path = 'encoded_dataset.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    autoencoder_model.to(device)
    if not only_process:
        encoded_dataset = encode(autoencoder_model, device, RESIZE)
        torch.save(encoded_dataset, 'encoded_dataset.pth')
    tStamp = datetime.now()
    for imPath in image_src:
        ftime = time.time()
        find_similar_images(autoencoder_model, en_dataset_path, imPath, tStamp, device)
        print(time.time() - ftime)


def load_autoencoder_model(model_path, autoencoder):
    model1 = autoencoder()
    model1.load_state_dict(torch.load(model_path))
    model1.eval()
    return model1


def plot_and_save(dataset_paths, input_image_path, similar_indices, timestamp):
    figures_folder = 'figures'
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)
    timestamp = timestamp.strftime("%Y%m%d%H%M%S")
    save_folder = os.path.join(figures_folder, f'figures_{timestamp}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    k = len(similar_indices)
    input_image = mpimg.imread(input_image_path)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, k + 1, 1)
    plt.imshow(input_image)
    plt.title('Input Image')
    plt.axis('off')
    _, tail = os.path.split(input_image_path)
    for i, idx in enumerate(similar_indices):
        similar_image_path = dataset_paths[idx]
        similar_image = mpimg.imread(similar_image_path)
        plt.subplot(1, k + 1, i + 2)
        plt.imshow(similar_image)
        plt.title(f'Similar {i + 1}')
        plt.axis('off')
    figure_path = os.path.join(save_folder, tail)
    plt.savefig(figure_path)
    plt.close()


def find_similar_images(autoencoder_model, encoded_dataset_path, input_image_path, timestamp, device, k=5):
    encoded_input = post_encode_image(autoencoder_model, input_image_path, RESIZE, device).flatten()
    encoded_dataset = load_encoded_dataset(encoded_dataset_path)
    encoded_dataset = encoded_dataset.reshape(encoded_dataset.shape[0], -1)
    similar_indices = knn_search(encoded_input, encoded_dataset, k)
    dataset_paths = load_dataset_paths()
    plot_and_save(dataset_paths, input_image_path, similar_indices, timestamp)


def knn_search(encoded_input, encoded_dataset, k=5):
    # Use kNN to find the most similar images
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(encoded_dataset)
    distances, indices = nbrs.kneighbors([encoded_input])
    return indices[0][1:]


def post_encode_image(model, image_path, resize, device):
    transform1 = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform1(image).unsqueeze(0).to(device)

    with torch.no_grad():
        encoded = model.encoder(image)

    return encoded.squeeze().cpu().numpy()
