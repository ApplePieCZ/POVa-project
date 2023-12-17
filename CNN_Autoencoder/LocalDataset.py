import torch
from torchvision import transforms as transforms
from Train import get_transform
from DataLoader import get_DataLoader


def encode(model_i, device_i, resize):
    model_i.eval()
    encoded_dataset_i = []
    i = 0
    transform = get_transform(resize)
    data_loader = get_DataLoader('test', transform, 1024, False)
    with torch.no_grad():
        for data_i in data_loader:
            print(i)
            i = i + 1
            images, _ = data_i
            images = images.to(device_i)

            # Encode the images using the encoder part of the autoencoder
            encoded_images = model_i.encoder(images)
            encoded_dataset_i.append(encoded_images)

    # Concatenate the list of encoded representations into a single numpy array
    encoded_dataset_i = torch.cat(encoded_dataset_i, dim=0)

    return encoded_dataset_i


def load_dataset_paths():
    transform = get_transform(256)
    data_loader = get_DataLoader('test', transform, 1, False)
    # Assuming your dataset consists of image files, get their paths
    dataset_paths = data_loader.dataset._image_files
    return dataset_paths


def load_encoded_dataset(dataset_path):
    # Load the encoded dataset (assuming it's a PyTorch tensor saved as a .pth file)
    encoded_dataset = torch.load(dataset_path)
    return encoded_dataset.cpu().numpy()
