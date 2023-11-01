import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import os
import scipy

model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))


def load_img(folder):
    print("Loading images")
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
            img_array = np.array(img)
            images.append(img_array)
    return np.array(images)


def calculate_fid(img1, img2):  # Ensure images are of the shape (299, 299, 3)

    # Preprocess images (scaling and other operations)
    images1 = preprocess_input(img1)
    images2 = preprocess_input(img2)


    print("Preprocessing images")

    # Calculate activations
    act1 = model(images1)
    act2 = model(images2)

    # Calculate mean and covariance statistics
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid


generated_img = load_img('generated_img')
real_img = load_img('real_img')

# Calculate FID
print("Calculating fid")
fid_value = calculate_fid(real_img, generated_img)
print(f"FID: {fid_value}")

assert fid_value < 1 == True;
