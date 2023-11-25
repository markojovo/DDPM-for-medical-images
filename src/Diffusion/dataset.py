from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms

class DiffusionDataset(Dataset):
    def __init__(self, directory, transform=None, num_images=None):
        self.total = 0
        self.directory = directory
        self.transform = transform if transform else transforms.ToTensor()

        # List all subfolders and optionally limit the number
        all_folders = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
        self.image_folders = all_folders if num_images is None else all_folders[:num_images]

        # Precompute the cumulative number of pairs in each folder
        self.cumulative_pairs = []
        for folder in self.image_folders:
            num_pairs = len(os.listdir(folder)) - 1
            self.total += num_pairs
            self.cumulative_pairs.append(self.total)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Find the folder that contains the image for this idx
        folder_idx = next(i for i, total in enumerate(self.cumulative_pairs) if total > idx)
        image_idx = idx - (self.cumulative_pairs[folder_idx - 1] if folder_idx > 0 else 0)

        folder_path = self.image_folders[folder_idx]
        input_image_path = os.path.join(folder_path, f"{str(image_idx + 1).zfill(3)}.jpg")
        target_image_path = os.path.join(folder_path, f"{str(image_idx).zfill(3)}.jpg")

        input_image = Image.open(input_image_path).convert('L')
        target_image = Image.open(target_image_path).convert('L')

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


# Example usage with limited number of images
# dataset = DiffusionDataset("../data/diffused_train/", transform=transforms.ToTensor(), num_images=10)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
