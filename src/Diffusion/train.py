import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from diffusion_model import StepwiseReverseDiffusionNet
from dataset import DiffusionDataset  # Import the custom dataset class
import torchvision.transforms as transforms
from util_functs import reconstruct_image_iteratively, clear_and_create_directory, save_reconstructed_images  # Import the function

def main():
    # Check for CUDA availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA is {'available' if torch.cuda.is_available() else 'not available'}. Using {'GPU' if torch.cuda.is_available() else 'CPU'}.")

    # Initialize the neural network and optimizer
    model = StepwiseReverseDiffusionNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Check if the model weights file exists and load it
    model_weights_path = 'diffusion_model.pth'
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path))
        print("Loaded saved model weights.")

    # Initialize custom dataset and DataLoader
    train_dataset = DiffusionDataset("../data/diffused_train/", num_images=100, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=320, num_workers = 8, shuffle=True)

    # Training loop
    epoch_losses = []
    total_epochs = 70
    save_interval = 250
    print("Starting training...")
    clear_and_create_directory("./training_plots/")

    for epoch in range(total_epochs):
        print(f"\nStarting Epoch {epoch + 1}/{total_epochs}...")
        epoch_loss_sum = 0.0
        num_batches = 0

        for batch_idx, (input_image, noise_levels, target_image) in enumerate(train_loader):
            print(f"  Processing batch {batch_idx + 1}/{len(train_loader)}... ",end='')
            input_image, noise_levels, target_image = input_image.to(device), noise_levels.to(device), target_image.to(device)

            # Forward pass with noise level
            output = model(input_image, noise_levels)
            loss = torch.nn.SmoothL1Loss()(output, target_image)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Batch loss: {loss.item()}...")
            epoch_loss_sum += loss.item()
            num_batches += 1


            epoch_avg_loss = epoch_loss_sum / num_batches
            epoch_losses.append(epoch_avg_loss)

            if (batch_idx % save_interval == 0):
                initial_noisy_images = torch.randn(5, 1, 128, 128).to(device)
                reconstructed_images = [reconstruct_image_iteratively(model, initial_noisy_image, 150).cpu().detach().numpy().squeeze(0) for initial_noisy_image in initial_noisy_images] # changed to 15 for testing

                save_reconstructed_images(epoch, batch_idx, reconstructed_images)

                # Save the model after every epoch
                torch.save(model.state_dict(), f'diffusion_model.pth')
                print(f"Model saved for Epoch {epoch+1}, batch {batch_idx}")


        print(f"Epoch {epoch + 1}/{total_epochs} completed, Average Loss: {epoch_avg_loss:.4f}")




    print("Training completed.")

    # Plotting the epoch-wise losses
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses)
    plt.title("Epoch-wise Average Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.show()

    # Inference and Plotting Reconstructed Images
    initial_noisy_images = torch.randn(5, 1, 128, 128).to(device)

    reconstructed_images = []
    for initial_noisy_image in initial_noisy_images:
        reconstructed_image = reconstruct_image_iteratively(model, initial_noisy_image, 150)
        reconstructed_images.append(reconstructed_image.cpu().detach().numpy().squeeze(0))
        
    plt.figure(figsize=(15, 3))
    for i, img in enumerate(reconstructed_images, 1):
        plt.subplot(1, 5, i)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')
    plt.suptitle('Reconstructed Images')
    plt.show()

if __name__ == '__main__':
    main()
