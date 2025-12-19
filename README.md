# Variational Autoencoder (VAE) Inference on MNIST  
**TensorFlow + Hugging Face Diffusers**

This repository demonstrates how to use a **pretrained Variational Autoencoder (VAE)** from the Hugging Face Diffusers library to **encode and reconstruct MNIST images** using **TensorFlow**.

The goal of this project is **conceptual clarity** — understanding how a real-world VAE works internally (encode → sample → decode) without training complexity.

---

## 🚀 What This Project Does

- Loads the MNIST dataset using Hugging Face `datasets`
- Preprocesses grayscale images to match Stable Diffusion VAE requirements
- Uses a **pretrained KL Autoencoder** (`stabilityai/sd-vae-ft-mse`)
- Encodes images into latent space
- Samples latent vectors using the reparameterization trick
- Decodes latents back into images
- Visualizes original vs reconstructed images

> ⚠️ This script runs the VAE **in inference mode only** — no training or fine-tuning.

---

## 🧠 Why MNIST with a Stable Diffusion VAE?

Although MNIST is simple, this setup helps you:
- Understand **real diffusion-style VAEs**
- Learn **latent distributions**, not just encoders
- Prepare for **diffusion models, image generation, and latent pipelines**

---

## 📦 Requirements

pip install tensorflow datasets diffusers matplotlib

## 🔍 Key Concepts Covered

Variational Autoencoders (VAE)

Latent distributions & sampling

Reparameterization trick

TensorFlow inference pipelines

Hugging Face Diffusers (TF backend)

## 🖼️ Output Example

The script displays:

Top row: Original MNIST images

Bottom row: Reconstructed images from the VAE

## 🧑‍🎓 Who Is This For?

Students learning VAEs

ML engineers transitioning to diffusion models

Anyone wanting TensorFlow + Hugging Face Diffusers examples

Researchers exploring latent-space representations

## 📌 Notes

Images are resized to 256×256 and converted to RGB

Pixel values are scaled to [-1, 1] (VAE requirement)

Uses Stable Diffusion’s VAE, not a toy model

## 📜 License

MIT License

## ⭐ If this helped you

Star the repo and share it with other ML learners!
