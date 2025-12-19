"""
Variational Autoencoder (VAE) Demo with TensorFlow and Hugging Face Diffusers

This script demonstrates how to use a pretrained KL Autoencoder (from Stability AI)
to encode and decode MNIST images. The workflow includes:

1. Loading the MNIST dataset via Hugging Face `datasets`.
2. Preprocessing images:
   - Convert grayscale to RGB
   - Resize to 256x256
   - Normalize pixel values to [-1, 1]
3. Creating a TensorFlow dataset pipeline for batching.
4. Using `TFAutoencoderKL` from Hugging Face Diffusers:
   - Encode images into latent space
   - Sample latent vectors with reparameterization
   - Decode back into reconstructed images
5. Visualizing original vs reconstructed images side by side with Matplotlib.

Note: The VAE is loaded in inference mode (`trainable=False`) to showcase encoding
and reconstruction rather than training from scratch.
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from diffusers import TFAutoencoderKL


# -----------------------------
# Load the MNIST dataset
# -----------------------------
# We load the 'train' split for demonstration; the dataset returns PIL images
# under the column name 'image'. We'll convert them to numeric arrays in
# preprocessing.
dataset = load_dataset("mnist", split="train")


def preprocess(example):
    """Preprocess a single example from the HF `datasets` object.

    The VAE we load expects 3-channel (RGB) images at 256x256 with pixel
    values in the range [-1, 1]. MNIST images are 28x28 grayscale. Steps:
    1. Convert PIL image → NumPy array with dtype float32 and scale to [0, 1].
    2. Stack channels to convert grayscale → RGB.
    3. Resize to 256x256 using a TF op (returns a TF tensor).
    4. Map pixel values to [-1, 1].

    The function returns a dict with a single key 'image' to keep compatibility
    with the HF `datasets` map API.
    """

    # Convert PIL.Image to NumPy array in [0, 1]
    image = np.array(example["image"]).astype("float32") / 255.0

    # MNIST is grayscale (shape: H x W). Convert to RGB by repeating channels
    # resulting shape: H x W x 3
    image = np.stack([image, image, image], axis=-1)

    # Use TensorFlow's resize for consistent pipeline behavior. The output is
    # a float32 Tensor shaped (256, 256, 3).
    image = tf.image.resize(image, (256, 256))

    # Map from [0, 1] to [-1, 1], which is commonly expected by image VAEs.
    image = image * 2.0 - 1.0

    # Return as a dictionary to match the HF datasets mapping convention.
    return {"image": image}


# Apply preprocessing to the whole dataset. Using the datasets API allows us
# to transform the examples before creating a TF dataset.
dataset = dataset.map(preprocess)

# Tell the `datasets` object to present data in a TensorFlow-friendly format
# so `dataset["image"]` yields objects that can be converted to TF tensors.
dataset.set_format(type="tensorflow", columns=["image"])


# -----------------------------
# Build a small TF data pipeline
# -----------------------------
batch_size = 4

# Create a tf.data.Dataset from the preprocessed image tensors. For a large
# dataset you would prefer `dataset.to_tf_dataset(...)` or use streaming.
train_ds = tf.data.Dataset.from_tensor_slices(dataset["image"])
train_ds = train_ds.batch(batch_size)


# -----------------------------
# Load pretrained VAE
# -----------------------------
# We load `stabilityai/sd-vae-ft-mse`, a VAE checkpoint often used with Stable
# Diffusion. `from_pt=False` tells the loader the weights are TF-compatible.
vae = TFAutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", from_pt=False)

# Set the VAE to inference mode to avoid any accidental training/weight updates.
vae.trainable = False


# -----------------------------
# Encode → Sample → Decode
# -----------------------------
# Take a single batch from our dataset to demonstrate the VAE's forward pass.
images = next(iter(train_ds))  # shape: (batch_size, 256, 256, 3), values in [-1,1]

# 1) Encode the images. The encoder produces a posterior object which contains
#    a distribution over latents (mean, variance) in `latent_dist`.
posterior = vae.encode(images)

# 2) Sample from the posterior distribution. `latent_dist` is typically a
#    `tfd.Normal`-like object (or a wrapper) exposing `sample()`. This is the
#    reparameterization step used in VAEs: z ~ q(z|x).
z = posterior.latent_dist.sample()

# 3) Decode the latent sample back to image space. The decoder returns an
#    object with `.sample` or direct tensors depending on implementation; here
#    we use `.sample` which is consistent with the library's TF wrapper.
reconstructed = vae.decode(z).sample


# -----------------------------
# Prepare images for visualization
# -----------------------------
# The VAE inputs/outputs are in [-1, 1]. Convert them to [0, 1] for plotting
# with Matplotlib and clamp to avoid artifacts from numerical imprecision.
images_vis = (images + 1.0) / 2.0
recon_vis = (reconstructed + 1.0) / 2.0

images_vis = tf.clip_by_value(images_vis, 0.0, 1.0)
recon_vis = tf.clip_by_value(recon_vis, 0.0, 1.0)


# -----------------------------
# Plot originals vs reconstructions
# -----------------------------
plt.figure(figsize=(8, 4))

for i in range(batch_size):
    # Top row: original images
    plt.subplot(2, batch_size, i + 1)
    plt.imshow(images_vis[i])
    plt.axis("off")
    if i == 0:
        plt.title("Original")

    # Bottom row: reconstructed images
    plt.subplot(2, batch_size, i + 1 + batch_size)
    plt.imshow(recon_vis[i])
    plt.axis("off")
    if i == 0:
        plt.title("Reconstructed")

plt.tight_layout()
plt.show()
