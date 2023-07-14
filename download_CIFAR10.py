import os
import torchvision
import torchvision.transforms as transforms

# Set the root directory where the dataset will be downloaded
data_root = "../dataset/"  # Replace with your desired data directory

# Define the transformations to apply to the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image pixels
])

# Download and load the CIFAR-10 training dataset
train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)

# Download and load the CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
