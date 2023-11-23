import numpy as np
import torch

def create_clean_multi_cross(size=128, thickness=6):
    """
    Function to create a clean pattern with multiple crosses and reduced thickness.
    Args:
    size (int): Size of the square pattern.
    thickness (int): Thickness of the cross lines.

    Returns:
    numpy.ndarray: A square pattern with crosses.
    """
    cross = np.zeros((size, size))
    mid = size // 2
    quarter = size // 4
    three_quarter = 3 * size // 4

    for pos in [quarter, mid, three_quarter]:
        cross[pos - thickness // 2:pos + thickness // 2, :] = 1
        cross[:, pos - thickness // 2:pos + thickness // 2] = 1
    return cross

def diffuse_image(image, noise_level=0.5):
    """
    Function to add noise to an image.
    Args:
    image (torch.Tensor): The image to add noise to.
    noise_level (float): The level of noise to add.

    Returns:
    torch.Tensor: The noisy image.
    """
    return image * (1 - noise_level) + torch.randn_like(image) * noise_level

def diffuse_image_levels_linear(image, levels):
    """
    Function to add noise to an image at multiple linear levels.
    Args:
    image (torch.Tensor): The image to add noise to.
    levels (int): Number of levels of noise to add.

    Returns:
    List[torch.Tensor]: List of images with increasing levels of noise.
    """
    noise_levels = np.linspace(0, 1, levels)
    return [diffuse_image(image, noise_level) for noise_level in noise_levels]

def diffuse_image_levels_sigmoid(image, levels, k=0.1, offset=0.5):
    """
    Function to add noise to an image at multiple sigmoidal levels.
    Args:
    image (torch.Tensor): The image to add noise to.
    levels (int): Number of levels of noise to add.
    k (float): The steepness of the sigmoid curve.
    offset (float): The offset of the sigmoid curve.

    Returns:
    List[torch.Tensor]: List of images with varying levels of noise.
    """
    noise_levels = [1 / (1 + np.exp(-k * (i - offset * levels))) for i in range(levels)]
    return [diffuse_image(image, noise_level) for noise_level in noise_levels]

def reconstruct_image_iteratively(model, initial_noisy_image, num_iterations):
    """
    Function to reconstruct an image iteratively using a given model.
    Args:
    model (torch.nn.Module): The neural network model to use for reconstruction.
    initial_noisy_image (torch.Tensor): The initial noisy image to start with.
    num_iterations (int): Number of iterations to perform.

    Returns:
    torch.Tensor: The reconstructed image.
    """
    reconstructed_image = initial_noisy_image.unsqueeze(0)  
    for _ in range(num_iterations):
        reconstructed_image = model(reconstructed_image)
    return reconstructed_image.squeeze(0)
