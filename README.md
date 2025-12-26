

# Multi-Device Deep Learning: CPU vs. GPU (CUDA) Performance Analysis

This repository contains a performance benchmarking project designed to compare training efficiencies between **Central Processing Units (CPU)** and **Graphics Processing Units (NVIDIA CUDA)**. Using a high-complexity "Ultra Deep" Neural Network built with **PyTorch**, the project demonstrates the significant speedups achieved through hardware acceleration in machine learning workflows.

## 🚀 Project Overview

The core objective of this project is to measure and visualize the computational advantages of using parallel processing (CUDA) over sequential processing (CPU). The project implements a robust deep-learning pipeline, including data preprocessing, model architecture design, and comparative evaluation metrics.

## 🧠 Model Architecture: "UltraDeepNN"

To ensure the hardware is adequately stressed for meaningful comparison, the project utilizes an `UltraDeepNN` class featuring:

* **8 Linear Layers:** Gradually tapering from 1024 to 64 hidden units.
* **GELU Activation Functions:** Using Gaussian Error Linear Units for smoother gradient flow.
* **Batch Normalization:** Applied after every linear layer to ensure stable training and faster convergence.
* **Dropout (0.3):** Strategic regularization to prevent overfitting in such a deep architecture.

## 🛠️ Key Features

* **Hybrid Device Support:** Automatically detects and utilizes **NVIDIA CUDA** or **Apple MPS** (Metal Performance Shaders) if available, defaulting to CPU otherwise.
* **Performance Benchmarking:** Real-time tracking of training time (seconds) and calculation of "Speedup %".
* **Comprehensive Evaluation:** Includes Accuracy scores, Weighted F1 scores, and Confusion Matrices to verify model consistency across devices.
* **Data Visualization:** Uses `Seaborn` and `Matplotlib` to generate bar plots for training time comparison and side-by-side confusion matrices.

## 📊 Performance Results

Based on the initial benchmarks conducted in a Google Colab environment:

* **CPU Training Time:** ~109.26 seconds.
* **CUDA Training Time:** ~12.51 seconds.
* **Total Speedup:** **88.55% faster** using GPU acceleration.
* **Model Accuracy:** Consistent ~59.45% across both devices, ensuring that hardware switching does not impact mathematical outcomes.

## 📁 Repository Structure

* `PDC Project (1).ipynb`: The main Jupyter notebook containing the implementation and results.


## ⚙️ Requirements

To run this project, you will need:

* Python 3.x
* PyTorch
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn

## 🔧 How to Use

1. **Clone the repository.**
2. **Upload your data:** Ensure `processed_dataset.csv` is in the root directory.
3. **Run the Notebook:** Execute the cells in `PDC Project (1).ipynb`. The script will automatically detect if you have a GPU enabled and run the comparison.

---

*Note: This project was developed and tested on Google Colab using a T4 GPU.*
