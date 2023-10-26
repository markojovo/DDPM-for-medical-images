# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Function to create a clean cross-like pattern
def create_clean_cross(size=64, thickness=4):
    cross = np.zeros((size, size))
    mid = size // 2
    cross[mid - thickness // 2:mid + thickness // 2, :] = 1
    cross[:, mid - thickness // 2:mid + thickness // 2] = 1
    return cross

# Function to add noise to an image
def diffuse_image(image, noise_level=0.5):
    return image * (1 - noise_level) + torch.randn_like(image) * noise_level

# Function to add noise to an image at multiple levels
def diffuse_image_levels(image, levels):
    noise_levels = np.linspace(0, 1, levels)
    return [diffuse_image(image, noise_level) for noise_level in noise_levels]

# Function to reconstruct an image step-by-step using the model
def reconstruct_image_stepwise(model, diffused_image_levels):
    reconstructed_image = diffused_image_levels[-1].clone().detach()
    for level_image in reversed(diffused_image_levels[:-1]):
        reconstructed_image = model(reconstructed_image).detach()
    return reconstructed_image

# Neural Network architecture
class StepwiseReverseDiffusionNet(nn.Module):
    def __init__(self):
        super(StepwiseReverseDiffusionNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Generate clean cross-like patterns for the dataset
num_samples = 25
X_train_clean = np.array([create_clean_cross() for _ in range(num_samples)])
X_train_clean = torch.tensor(X_train_clean, dtype=torch.float32).unsqueeze(1)

# Generate diffused images at different levels (4 levels in one of the steps)
levels = 1  # Changed based on steps
diffused_images_levels = [diffuse_image_levels(img, levels) for img in X_train_clean]
diffused_images_levels = torch.cat([torch.stack(level) for level in zip(*diffused_images_levels)], dim=0)

# Initialize the neural network and optimizer
model = StepwiseReverseDiffusionNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 2  # Changed based on steps
losses = []
for epoch in range(epochs):
    for i in range(num_samples * levels):
        original_image = X_train_clean[i // levels:i // levels + 1]
        diffused_image = diffused_images_levels[i:i + 1]
        output = model(diffused_image)
        loss = nn.MSELoss()(output, original_image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
