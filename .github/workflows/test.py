import math
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


from scipy.stats import entropy

#
# def calculate_inception_score(images, n_split=10, eps=1E-16):
#     # Assume images are numpy arrays with shape (n_images, height, width, channels)
#     tf.config.run_functions_eagerly(True)
#
#     scores = []
#     n_part = int(np.floor(images.shape[0] / n_split))
#
#     for i in range(n_split):
#         subset = images[i * n_part:(i + 1) * n_part]
#         subset = preprocess_input(subset)
#         pred = model.predict(subset)
#         kl_div = pred * (np.log(pred + eps) - np.log(np.expand_dims(np.mean(pred, axis=0), axis=0)))
#         kl_div = np.mean(np.sum(kl_div, axis=1))
#         scores.append(np.exp(kl_div))
#
#     return np.mean(scores), np.std(scores)


from sklearn.manifold import TSNE
from scipy.stats import entropy

def calculate_fds(images_real, images_fake, epsilon=1e-6):
    # Preprocess and flatten images
    images_real = preprocess_input(images_real.reshape((images_real.shape[0], -1)))
    images_fake = preprocess_input(images_fake.reshape((images_fake.shape[0], -1)))

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=3, random_state=42)  # Adjust perplexity as needed
    real_tsne = tsne.fit_transform(images_real)
    fake_tsne = tsne.fit_transform(images_fake)

    # Fit Gaussian models
    real_mean, real_cov = np.mean(real_tsne, axis=0), np.cov(real_tsne, rowvar=False)
    fake_mean, fake_cov = np.mean(fake_tsne, axis=0), np.cov(fake_tsne, rowvar=False)

    # Regularize covariance matrices
    real_cov += np.eye(real_cov.shape[0]) * epsilon
    fake_cov += np.eye(fake_cov.shape[0]) * epsilon

    # Calculate KL Divergence
    fds = entropy(pk=real_cov, qk=fake_cov)

    return fds



from skimage.metrics import structural_similarity as ssim


def calculate_diversity_score(images, images2, num_comparisons):
    images_gray = np.dot(images[..., :3], [0.2989, 0.5870, 0.1140])
    images2_gray = np.dot(images2[..., :3], [0.2989, 0.5870, 0.1140])

    ssim_values = []
    for i in range(num_comparisons):
        idx1, idx2 = np.random.choice(images_gray.shape[0], 2, replace=False)
        score, _ = ssim(images_gray[idx1], images_gray[idx2], full=True, channel_axis=-1, data_range=images_gray.max() - images_gray.min())
        ssim_values.append(score)

    real_ssim_values = calculate_ssim_distribution(images2_gray, num_comparisons)

    ds = entropy(pk=ssim_values, qk=real_ssim_values)

    return ds



from skimage.metrics import structural_similarity as ssim

def calculate_ssim_distribution(images, num_comparisons):
    ssim_scores = []
    for i in range(num_comparisons):
        idx1, idx2 = np.random.choice(images.shape[0], 2, replace=False)
        score = ssim(images[idx1], images[idx2], channel_axis=-1, data_range=images.max() - images.min())
        ssim_scores.append(score)
    return ssim_scores


def test_fid():
    generated_img = load_img('.github/workflows/generated_img')
    real_img = load_img('.github/workflows/real_img')
    # Calculate FID
    print("Calculating fid")
    fid_value = calculate_fid(real_img, generated_img)
    print(f"FID: {fid_value}")
    assert (fid_value < math.inf) == True

# def test_is():
#     generated_img = load_img('.github/workflow/generated_img')
#     # Calculate IS
#     print("Calculating IS")
#     is_mean, is_std = calculate_inception_score(generated_img)
#     print(f"IS: {is_mean} ± {is_std}")
#     assert is_mean > 0

def test_fds():
    generated_img = load_img('.github/workflows/generated_img')
    real_img = load_img('.github/workflows/real_img')
    print("Calculating FDS")
    fds_value = calculate_fds(real_img, generated_img)
    print(f"FDS: {fds_value}")
    assert fds_value.all() >= 0

def test_ds():
    generated_img = load_img('.github/workflows/generated_img')
    real_img = load_img('.github/workflows/real_img')

    num_comparisons = min(len(generated_img), len(real_img)) * (min(len(generated_img), len(real_img)) - 1) // 2

    print("Calculating DS")
    ds_value = calculate_diversity_score(generated_img, real_img, num_comparisons)
    print(f"DS: {ds_value}")
    assert ds_value >= 0

test_fid()
test_fds()
test_ds()
