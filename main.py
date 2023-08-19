import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
from vae_simple import *


def train(model_, optimizer_, epochs, device_):
    model_.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, 28 * 28).to(device_)

            optimizer_.zero_grad()

            x_hat, mean, log_var = model_(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer_.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))
    return overall_loss


def plot_latent_space(model_, scale=1.0, n=25, digit_size=28, fig_size_=15):
    # display an n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model_.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size, ] = digit

    plt.figure(figsize=(fig_size_, fig_size_))
    plt.title('VAE Latent Space Visualization')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


if __name__ == "__main__":
    # create a transform to apply to each datapoint
    transform = transforms.Compose([transforms.ToTensor()])

    # download the MNIST datasets
    path = '~/datasets'
    train_dataset = MNIST(path, transform=transform, download=True)
    test_dataset = MNIST(path, transform=transform, download=True)

    # create train and test dataloaders
    batch_size = 100
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get 25 sample training images for visualization
    dataiter = iter(train_loader)
    image = next(dataiter)

    num_samples = 25
    sample_images = [image[0][i, 0] for i in range(num_samples)]

    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

    for ax, im in zip(grid, sample_images):
        ax.imshow(im, cmap='gray')
        ax.axis('off')

    plt.show()

    # Train
    """
    model = VAE(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, optimizer, epochs=50, device_=device)
    
    # save
    torch.save(model.state_dict(), 'mnist.pth')
    """

    # Load and Infer
    model = VAE(device=device)
    model.load_state_dict(torch.load('mnist.pth'))
    model.eval()

    plot_latent_space(model)
