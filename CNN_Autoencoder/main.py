# --------------------------------------------------------------------------------------------------------------
# Code skeleton based on https://www.geeksforgeeks.org/implement-convolutional-autoencoder-in-pytorch-with-cuda/
# --------------------------------------------------------------------------------------------------------------
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import image as mpimg
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
RESIZE = 64
EPOCHS = 25
TRAIN = False
ONLY_PROCESS = False
IMPATH= r"flowers\flowers-102\jpg\image_00001.jpg"
SOURCE_IMGS=[r""]
TRAINVALBATCH = 128
STOREBATCH = 1024*64//RESIZE

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_autoencoder_model(model_path):
    model1 = Autoencoder()
    model1.load_state_dict(torch.load(model_path))
    model1.eval()
    return model1


def encode_whole_dataset(model_i, data_loader, device_i):
    model_i.eval()
    encoded_dataset_i = []
    i = 0
    with torch.no_grad():
        for data_i in data_loader:
            print(i)
            i = i + 1
            images, _ = data_i
            images = images.to(device_i)

            
            encoded_images = model_i.encoder(images)
            encoded_dataset_i.append(encoded_images)

    
    encoded_dataset_i = torch.cat(encoded_dataset_i, dim=0)

    return encoded_dataset_i


def encode_image(model, image_path):
    transform1 = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform1(image).unsqueeze(0)

    with torch.no_grad():
        encoded = model.encoder(image)

    return encoded.squeeze().numpy()


def load_encoded_dataset(dataset_path):
    
    encoded_dataset = torch.load(dataset_path)
    return encoded_dataset.cpu().numpy()


def knn_search(encoded_input, encoded_dataset, k=5):
    
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(encoded_dataset)
    distances, indices = nbrs.kneighbors([encoded_input])
    return indices[0][1:]


def find_similar_images(model, dataset_path, encoded_dataset_path, input_image_path, timestamp, k=5):
    
    autoencoder_model = load_autoencoder_model('conv_autoencoder.pth')

    
    encoded_input = encode_image(autoencoder_model, input_image_path)
    encoded_input = encoded_input.flatten()
    
    encoded_dataset = load_encoded_dataset(encoded_dataset_path)
    encoded_dataset = encoded_dataset.reshape(encoded_dataset.shape[0], -1)
    
    similar_indices = knn_search(encoded_input, encoded_dataset, k)

    
    dataset_paths = sorted([os.path.join(dataset_path, file) for file in os.listdir(dataset_path)])

    figures_folder = 'figures'
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

    
    timestamp = timestamp.strftime("%Y%m%d%H%M%S")
    save_folder = os.path.join(figures_folder, f'figures_{timestamp}')

    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    
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


if TRAIN:
    
    model = Autoencoder()

    
    transform = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor(),
    ])

    
    train_dataset = datasets.Flowers102(root='flowers',
                                        split='train',
                                        transform=transform,
                                        download=True)
    test_dataset = datasets.Flowers102(root='flowers',
                                       split='test',
                                       transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=TRAINVALBATCH,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=TRAINVALBATCH,
                                              shuffle=True)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    encoded_dataset = None
    
    num_epochs = EPOCHS
    output = None
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    if output is not None:
        encoded_dataset = output.cpu().detach().numpy()

    
    torch.save(model.state_dict(), 'conv_autoencoder.pth')
    torch.save(torch.tensor(encoded_dataset), 'encoded_dataset.pth')

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon = model(data)
            break

    plt.figure(dpi=250)
    fig, ax = plt.subplots(2, 7, figsize=(15, 4))
    for i in range(7):
        ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
        ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
        ax[0, i].axis('OFF')
        ax[1, i].axis('OFF')
    plt.show()
else:
    autoencoder_model = load_autoencoder_model('conv_autoencoder.pth')
    dataset_path = r'flowers'
    en_dataset_path = 'encoded_dataset.pth'
    input_image_path = IMPATH
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder_model.to(device)
    if not ONLY_PROCESS:
        
        transform = transforms.Compose([
            transforms.Resize((RESIZE, RESIZE)),
            transforms.ToTensor(),
        ])

        
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        data_loader = DataLoader(dataset, batch_size=STOREBATCH, shuffle=False)

        
        encoded_dataset = encode_whole_dataset(autoencoder_model, data_loader, device)

        
        torch.save(encoded_dataset, 'encoded_dataset.pth')
        print("Encoded dataset saved successfully.")
    tStamp = datetime.now()
    for imPath in SOURCE_IMGS:
        find_similar_images(autoencoder_model, dataset_path + r"\flowers-102\jpg", en_dataset_path, imPath, tStamp)
