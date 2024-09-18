import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    # Create a new instance of the class
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the dataset with mean and std deviation
    ])

    # Download and load the training dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Download and load the test dataset
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Function to show an image
    def imshow(img):
        print(img.shape)
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
        plt.show()

    # Get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Show images
    imshow(torchvision.utils.make_grid(images))
    # Print labels
    print(' '.join('%5s' % labels[j].item() for j in range(8)))

    print("MNIST dataset is ready for training with PyTorch.")


