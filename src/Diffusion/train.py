import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from diffusion_model import StepwiseReverseDiffusionNet
from util_functs import create_clean_multi_cross, diffuse_image_levels_linear, reconstruct_image_iteratively

# Check for CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA is {'available' if torch.cuda.is_available() else 'not available'}. Using {'GPU' if torch.cuda.is_available() else 'CPU'}.")

# Generate clean multi-cross patterns for the dataset
num_samples = 5
X_train_multi_clean = np.array([create_clean_multi_cross() for _ in range(num_samples)])
X_train_multi_clean = torch.tensor(X_train_multi_clean, dtype=torch.float32).unsqueeze(1).to(device)

# Generate diffused images at different linear levels
levels = 1000
diffused_images_levels_linear = [diffuse_image_levels_linear(img, levels) for img in X_train_multi_clean]
diffused_images_levels_linear = torch.cat([torch.stack(level) for level in zip(*diffused_images_levels_linear)], dim=0)
diffused_images_levels_linear = diffused_images_levels_linear.to(device)

# Initialize the neural network and optimizer
model = StepwiseReverseDiffusionNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Prepare training data
flat_diffused = diffused_images_levels_linear.view(-1, 1, 128, 128)
pairs = [(flat_diffused[i], flat_diffused[i-1]) for i in range(1, flat_diffused.shape[0]) if i % 100 != 0]
inputs, targets = zip(*pairs)
inputs = torch.stack(inputs)
targets = torch.stack(targets)

train_dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Training loop
epoch_losses = []
total_epochs = 5
for epoch in range(total_epochs):
    epoch_loss_sum = 0.0
    num_batches = 0

    for original_images, diffused_images in train_loader:
        original_images, diffused_images = original_images.to(device), diffused_images.to(device)
        output = model(diffused_images)
        loss = torch.nn.SmoothL1Loss()(output, original_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss_sum += loss.item()
        num_batches += 1

    epoch_avg_loss = epoch_loss_sum / num_batches
    epoch_losses.append(epoch_avg_loss)
    print(f"Epoch [{epoch+1}/{total_epochs}], Average Loss: {epoch_avg_loss:.4f}")

# Plotting the epoch-wise losses
plt.figure(figsize=(10, 5))
plt.plot(epoch_losses)
plt.title("Epoch-wise Average Loss")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.show()

# Inference and Plotting Reconstructed Images
# Generate random noise inputs
initial_noisy_images = torch.randn(5, 1, 128, 128).to(device)

# Run inference and collect the reconstructed images
reconstructed_images = []
for initial_noisy_image in initial_noisy_images:
    reconstructed_image = reconstruct_image_iteratively(model, initial_noisy_image, 100)
    reconstructed_images.append(reconstructed_image.cpu().detach().numpy())

# Plot the reconstructed images
plt.figure(figsize=(15, 3))
for i, img in enumerate(reconstructed_images, 1):
    plt.subplot(1, 5, i)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.axis('off')
plt.suptitle('Reconstructed Images')
plt.show()
