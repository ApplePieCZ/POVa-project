import torch
from matplotlib import pyplot as plt
from torch import nn as nn, optim as optim
from torchvision import transforms as transforms, datasets as datasets
from DataLoader import get_DataLoader

EPOCHS = 25
RESIZE = 32
TRAINVALBATCH = 32


def get_transform(resize):
    return transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
    ])


def train_model(model, train_loader, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    encoded_dataset = None
    output = None

    for epoch in range(EPOCHS):
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, EPOCHS, loss.item()))

    if output is not None:
        encoded_dataset = output.cpu().detach().numpy()
    torch.save(model.state_dict(), 'conv_autoencoder.pth')
    torch.save(torch.tensor(encoded_dataset), 'encoded_dataset.pth')

    return model, encoded_dataset


def visualize_results(test_loader, model, device):
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


def train(autoencoder):
    model = autoencoder()
    transform = get_transform(RESIZE)
    train_loader = get_DataLoader('train', transform, TRAINVALBATCH, True)
    test_loader = get_DataLoader('test', transform, TRAINVALBATCH, True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, encoded_dataset = train_model(model, train_loader, device)
    visualize_results(test_loader, model, device)
