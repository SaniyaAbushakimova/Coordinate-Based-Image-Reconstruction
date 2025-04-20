# Coordinate-Based-Image-Reconstruction

Project completed on March 13, 2025.

## Project Overview

This project explores how **coordinate-based neural networks** can be used to reconstruct images from scratch—using only the **(x, y) pixel coordinates** as inputs and learning to predict the corresponding **(R, G, B)** color values. Instead of treating an image as a grid of pixel values, the model learns a **continuous function** that maps spatial coordinates to color. The goal is to investigate how well a neural network can **memorize and reproduce an image** when trained only on location-based inputs.

Inspired by the paper: “Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains”
Tancik et al., *NeurIPS 2020* ([paper link](https://arxiv.org/pdf/2006.10739)).

Tancik et al. (NeurIPS 2020) introduced Fourier feature mappings to address spectral bias in neural networks, this project implements and experiments with various input encoding strategies to evaluate their effect on image reconstruction quality.

## What This Project Includes

* A fully connected multi-layer neural network implemented from scratch using NumPy.

* Multiple input encoding strategies to transform the input (x, y):
  * **No mapping**: $\gamma(\mathbf{v})= \mathbf{v}$.
  * **Basic mapping**: $\gamma(\mathbf{v})=\left[ \cos(2 \pi \mathbf{v}),\sin(2 \pi \mathbf{v}) \right]^\mathrm{T}$.
  * **Gaussian Fourier feature mapping**: $\gamma(\mathbf{v})= \left[ \cos(2 \pi \mathbf B \mathbf{v}), \sin(2 \pi \mathbf B \mathbf{v}) \right]^\mathrm{T}$,
where each entry in $\mathbf B \in \mathbb R^{m \times d}$ is sampled from $\mathrm{N}(0, \sigma^2)$.

* A complete training and evaluation pipeline for:
  * **Low-resolution** (32x32) image reconstruction.
  * **High-resolution** (128x128) image memorization with sharp detail recovery.

* Extensive experimentation with:
  * Network depth and hidden size.
  * Learning rates and optimizers (mini-batch SGD, Adam).
  * Loss functions (MSE, MAE, Huber).
  * Fourier feature size (m) and scale (σ).
  * Image inpainting and reconstruction using masked data.

This project blends **deep learning**, **signal processing**, and **function approximation theory**, while reinforcing core concepts like backpropagation, gradient checking, and model optimization — all without relying on high-level ML libraries like PyTorch or TensorFlow.

## Results Summary

**1. Low-resolution image**

*Optimizer: mini-batch SGD, Loss: MSE*
<img width="1164" alt="Pasted Graphic 4" src="https://github.com/user-attachments/assets/2bf417ea-0511-409a-8df4-bd2c57c2c6e8" />

*Optimizer: mini-batch SGD, Losses: MAE, MSE, Huber, Mapping: Gaussian Fourier*
<img width="1162" alt="Pasted Graphic 8" src="https://github.com/user-attachments/assets/30548823-890d-4427-aa41-3c9cce6150e5" />

*Optimizer: Adam, Loss: MSE*
<img width="1162" alt="Pasted Graphic 6" src="https://github.com/user-attachments/assets/33c11e21-40e8-405b-b7eb-902c2f2b868f" />



**2. High-resolution image**

*Optimizer: mini-batch SGD, Loss: MSE*
<img width="1164" alt="Pasted Graphic 5" src="https://github.com/user-attachments/assets/bacf6563-2b4e-433e-9908-bd702f27bb49" />

*Optimizer: Adam, Loss: MSE*
<img width="1162" alt="Pasted Graphic 7" src="https://github.com/user-attachments/assets/7aac9dd2-d42d-4650-b115-8c498542b624" />

**3. Image inpainting**

<img width="1368" alt="image" src="https://github.com/user-attachments/assets/b7e907a1-5534-4490-b3f0-79b1e8371198" />


## Repository Contents

* `models/neural_net.py`
Contains the full multi-layer neural network implementation from scratch using NumPy. This includes forward and backward passes, parameter updates, and support for multiple layers and activation functions.

* `neural_network.ipynb` -- The main notebook for running all experiments. This serves as the core training and evaluation pipeline for reconstructing images using coordinate-based neural networks. It integrates model setup, data preprocessing, input feature mappings, training loops, and visualization of results.

* `develop_neural_network.ipynb` -- A standalone notebook designed for debugging and validating the neural network implementation using a toy dataset. It performs gradient checking to ensure that backpropagation is implemented correctly—a critical step before running full-scale experiments.

* `utils/gradient_check.py` -- Utility functions for performing numerical gradient checking. Used inside develop_neural_network.ipynb to compare analytical gradients with numerical approximations and verify correctness.

* `Report.pdf` -- A detailed summary of all experimental results, design choices, and observations. Includes comparisons across different input mappings (none, basic, Gaussian Fourier), hyperparameter tuning for low- and high-resolution reconstructions, optimizer evaluations (SGD vs Adam), loss function comparisons (MSE, MAE, Huber), and extended experiments such as deeper networks and image inpainting.
