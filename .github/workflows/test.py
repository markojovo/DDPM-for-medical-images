import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Load InceptionV3 model pre-trained on ImageNet data and exclude top classification layer
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

def calculate_fid(img1, img2):
    # Ensure images are of the shape (299, 299, 3)
    img1 = tf.image.resize(img1, (299, 299))
    img2 = tf.image.resize(img2, (299, 299))

    # Preprocess images (scaling and other operations)
    img1 = preprocess_input(img1)
    img2 = preprocess_input(img2)

    print("Preprecessing images")

    # Calculate activations
    act1 = model(img1)
    act2 = model(img2)

    # Calculate mean and variance statistics
    mu1, sigma1_sq = act1.numpy().mean(), act1.numpy().var()
    mu2, sigma2_sq = act2.numpy().mean(), act2.numpy().var()

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)

    fid = ssdiff + sigma1_sq + sigma2_sq - 2.0 * np.sqrt(sigma1_sq * sigma2_sq)

    return fid

# Load your images (replace with your paths or image loading method)
real_image = tf.keras.preprocessing.image.load_img('real.png', target_size=(299, 299))
real_image = np.array(real_image)

generated_image = tf.keras.preprocessing.image.load_img('test.png', target_size=(299, 299))
generated_image = np.array(generated_image)

# Images should be in the shape (N, 299, 299, 3), so we need to expand dimensions
real_image = np.expand_dims(real_image, axis=0)
generated_image = np.expand_dims(generated_image, axis=0)

# Calculate FID
print("Calculating fid")
fid_value = calculate_fid(real_image, generated_image)
print(f"FID: {fid_value}")

assert fid_value<0.01 == True;