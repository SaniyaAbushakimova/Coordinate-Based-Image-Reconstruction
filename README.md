# Coordinate-Based-Image-Reconstruction

Project completed on March 13, 2025.

## Project Overview


## Results Summary


## Repository Contents

* `models/neural_net.py`
Contains the full multi-layer neural network implementation from scratch using NumPy. This includes forward and backward passes, parameter updates, and support for multiple layers and activation functions.

* `neural_network.ipynb` -- The main notebook for running all experiments. This serves as the core training and evaluation pipeline for reconstructing images using coordinate-based neural networks. It integrates model setup, data preprocessing, input feature mappings, training loops, and visualization of results.

* `develop_neural_network.ipynb` -- A standalone notebook designed for debugging and validating the neural network implementation using a toy dataset. It performs gradient checking to ensure that backpropagation is implemented correctly—a critical step before running full-scale experiments.

* `utils/gradient_check.py` -- Utility functions for performing numerical gradient checking. Used inside develop_neural_network.ipynb to compare analytical gradients with numerical approximations and verify correctness.

* `Report.pdf` -- A detailed summary of all experimental results, design choices, and observations. Includes comparisons across different input mappings (none, basic, Gaussian Fourier), hyperparameter tuning for low- and high-resolution reconstructions, optimizer evaluations (SGD vs Adam), loss function comparisons (MSE, MAE, Huber), and extended experiments such as deeper networks and image inpainting.
