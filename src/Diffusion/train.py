import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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

    # Initialize custom dataset and DataLoader
    train_dataset = DiffusionDataset("../data/diffused_train/", transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=320, shuffle=True, num_workers=8)

    # Training loop
    epoch_losses = []
    total_epochs = 250
    save_interval = 1
    print("Starting training...")
    clear_and_create_directory("./training_plots/")

    for epoch in range(total_epochs):
        print(f"\nStarting Epoch {epoch + 1}/{total_epochs}...")
        epoch_loss_sum = 0.0
        num_batches = 0

        for batch_idx, (diffused_image, original_image) in enumerate(train_loader):
            print(f"  Processing batch {batch_idx + 1}/{len(train_loader)}...")
            diffused_image, original_image = diffused_image.to(device), original_image.to(device)

            # Forward pass
            output = model(diffused_image)
            loss = torch.nn.SmoothL1Loss()(output, original_image)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            num_batches += 1

            epoch_avg_loss = epoch_loss_sum / num_batches
            epoch_losses.append(epoch_avg_loss)


        print(f"Epoch {epoch + 1}/{total_epochs} completed, Average Loss: {epoch_avg_loss:.4f}")


        if (epoch % save_interval == 0):
            initial_noisy_images = torch.randn(5, 1, 128, 128).to(device)
            reconstructed_images = [reconstruct_image_iteratively(model, initial_noisy_image, 50).cpu().detach().numpy().squeeze(0) for initial_noisy_image in initial_noisy_images]

            save_reconstructed_images(epoch, reconstructed_images)

            # Save the model after every epoch
            torch.save(model.state_dict(), f'diffusion_model.pth')
            print(f"Model saved for Epoch {epoch+1}")


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
        reconstructed_image = reconstruct_image_iteratively(model, initial_noisy_image, 50)
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
