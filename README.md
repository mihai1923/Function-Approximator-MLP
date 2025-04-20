# Octave-MLP-Framework  

## Overview
This project implements fully-connected neural networks (Multi-Layer Perceptrons - MLPs) from scratch using GNU Octave. It provides modules for classification, regression, and time series forecasting tasks.

## Neural Network Implementations

### 1. Classification Neural Network
*   **Implementation:** An MLP designed for classification tasks.
*   **Core Logic:** Implements forward and backward propagation for training.
*   **Activation Functions:** Utilizes functions like Swish (or ReLU, Sigmoid, Tanh) for hidden layers and Softmax for the output layer, enabling multi-class probability outputs.
*   **Loss Function:** Employs Cross-Entropy loss, suitable for classification problems.
*   **Capabilities:** Can learn complex non-linear decision boundaries, demonstrated on tasks like the spiral dataset classification. Includes feature engineering capabilities (e.g., adding polar coordinates).

### 2. Regression Neural Network
*   **Implementation:** An MLP tailored for function approximation (regression).
*   **Core Logic:** Implements forward and backward propagation.
*   **Activation Functions:** Typically uses Tanh or other suitable activations for hidden layers and a Linear (Identity) activation for the output layer to predict continuous values.
*   **Loss Function:** Uses Mean Squared Error (MSE) loss to measure prediction accuracy.
*   **Capabilities:** Approximates mathematical functions, including user-defined custom functions. Implements segmented regression, where the input range is divided, and separate models are trained for each segment to handle complex functions more effectively. May utilize adaptive network architectures based on analyzed function complexity.

### 3. Time Series Forecasting Network
*   **Implementation:** An MLP adapted for sequence prediction and time series forecasting.
*   **Core Logic:** Implements forward and backward propagation, often using sequence data as input.
*   **Activation Functions:** Commonly uses Tanh for hidden layers and a Linear (Identity) activation for the output layer.
*   **Capabilities:** Forecasts future values based on past sequences. Supports custom time series function definitions. Includes implementations for frequency-aware training to better handle oscillatory or seasonal patterns and segmented forecasting approaches.

## Implementation Details (Built in Octave)

### Core Neural Network Components
*   **Network Initialization:** Function to create the network structure (weights, biases) based on specified layer sizes (`initNeuralNetwork.m`).
*   **Forward Pass:** Calculates activations layer by layer from input to output (`forward_pass.m`).
*   **Backward Pass:** Computes gradients of the loss function with respect to weights and biases using backpropagation (`backward_pass.m`).
*   **Prediction:** Uses the trained model to generate outputs for new inputs (`predict.m`).
*   **Loss Calculation:** Computes the network's loss (`compute_loss.m`, `mean_squared_error.m`, `cross_entropy.m`).
*   **Accuracy Calculation:** Computes classification accuracy (`compute_accuracy.m`).

### Activation Functions
*   Sigmoid (`sigmoid.m`, `sigmoid_prime.m`)
*   Tanh (`tanh.m`, `tanh_prime.m`)
*   ReLU (`relu.m`, `relu_prime.m`)
*   Leaky ReLU (`leaky_relu.m`, `leaky_relu_prime.m`)
*   Softmax (`softmax.m`, `softmax_prime.m`) - For classification output
*   Identity/Linear (`identity.m`, `identity_prime.m`) - For regression/forecasting output
*   Swish (Implemented directly where used)

### Loss Functions
*   Mean Squared Error (MSE) (`mean_squared_error.m`, `mean_squared_error_prime.m`) - For regression/forecasting
*   Cross-Entropy (`cross_entropy.m`, `cross_entropy_prime.m`) - For classification

### Training & Optimization
*   **Gradient Descent:** Standard, Stochastic (Mini-batch).
*   **Optimizers:** Includes implementations for SGD with Momentum and potentially Adam (within `train_enhanced.m` or similar).
*   **Learning Rate Schedules:** Implementations for adaptive learning rates and decay (e.g., Cosine Annealing, step decay within training functions).
*   **Early Stopping:** Monitors validation loss to prevent overfitting and stop training when improvement ceases.
*   **Batch Normalization:** Implemented to stabilize training and improve convergence (integrated within training/pass functions).
*   **Gradient Checking:** Utility to numerically verify the correctness of the backpropagation implementation (`check_gradients.m`).

### Regularization Techniques
*   **L2 Regularization:** Added to the loss function to penalize large weights.
*   **Dropout:** Implemented during training to randomly zero out activations, preventing co-adaptation.
*   **Gradient Clipping:** Limits the magnitude of gradients to prevent exploding gradients.
*   **Weight Constraints:** Techniques like Max Norm applied to limit the size of weights (`apply_weight_constraints.m` likely called within training).

### Data Handling & Feature Engineering
*   **Data Generation:** Scripts to create spiral datasets, custom regression datasets, and custom time series data (`generate_*.m` files).
*   **Normalization:** Standard scaling (mean/std deviation) of input features.
*   **Feature Engineering:** Example implementation includes adding polar coordinates (r, theta) and their powers for the spiral dataset.
*   **Segmentation:** Logic for dividing data ranges or time series for specialized model training (`run_segmented_*.m`, `segmented_*.m`).

## Usage Guide

### Running the Examples
The primary way to run the different network types and see examples is via the main script:
```octave
octave run_examples.m
```
This script provides options to choose between classification, regression, and time series forecasting tasks, often including sub-options for different datasets or custom functions.

## Real-World Applications

While this project serves as an educational tool to understand neural networks from the ground up, the concepts and types of networks implemented have numerous real-world applications:

### Classification Applications
*   **Image Recognition:** Identifying objects, faces, or scenes in images (e.g., automated photo tagging).
*   **Spam Detection:** Classifying emails as spam or not spam.
*   **Medical Diagnosis:** Assisting doctors by classifying medical images (like X-rays or MRIs) or patient data to detect potential diseases.
*   **Sentiment Analysis:** Determining the sentiment (positive, negative, neutral) expressed in text (e.g., product reviews, social media posts).

### Regression Applications
*   **Predictive Modeling:** Estimating continuous values like house prices based on features (size, location), predicting customer lifetime value, or forecasting energy consumption.
*   **Financial Forecasting:** Predicting stock prices or market trends (though highly complex and often requiring more advanced models).
*   **Scientific Modeling:** Approximating complex physical or biological processes based on input parameters.
*   **Demand Forecasting:** Predicting future demand for products or services.

### Time Series Forecasting Applications
*   **Weather Forecasting:** Predicting future temperature, rainfall, or other meteorological conditions based on historical data.
*   **Financial Market Analysis:** Forecasting stock prices, exchange rates, or commodity prices based on past trends.
*   **Sales Forecasting:** Predicting future sales volumes for businesses.
*   **Resource Management:** Forecasting demand for resources like electricity, water, or bandwidth.
*   **Anomaly Detection:** Identifying unusual patterns in sequential data, such as detecting fraudulent transactions or equipment failures.

This implementation provides a foundational understanding necessary to tackle these more complex, real-world problems using neural networks.