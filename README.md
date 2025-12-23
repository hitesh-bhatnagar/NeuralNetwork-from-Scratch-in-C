# Neural Network from Scratch in C

[![Language](https://img.shields.io/badge/Language-C-blue.svg)](https://en.wikipedia.org/wiki/C_(programming_language))
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, dependency-free implementation of a **Multi-Layer Perceptron (MLP)** written entirely in C. This project demonstrates the core mechanics of machine learning—forward propagation, backpropagation, and gradient descent—without using high-level libraries like TensorFlow or PyTorch.

## Overview

The network is trained to solve the **XOR (Exclusive OR)** problem. XOR is a classic challenge in AI history because it is not linearly separable, meaning a simple single-layer perceptron cannot solve it. This implementation uses a hidden layer with non-linear activation to successfully learn the XOR logic.

### Key Features
- **Pure C implementation**: Uses only `<stdio.h>`, `<stdlib.h>`, and `<math.h>`.
- **Backpropagation**: Manually derived partial derivatives for weight updates.
- **Sigmoid Activation**: Smooth, non-linear squash function for neurons.
- **Dynamic Shuffling**: Implements the Fisher-Yates shuffle to improve SGD convergence.

## Architecture

The model uses a **2-2-1** architecture:
- **Input Layer**: 2 neurons (Input $A$ and $B$)
- **Hidden Layer**: 2 neurons with Sigmoid activation.
- **Output Layer**: 1 neuron with Sigmoid activation.



### The Math Behind it
The network minimizes error using the **Chain Rule**.
1. **Forward Pass**: 
   $$a = \sigma(\sum (w \cdot i) + b)$$
2. **Error Calculation**: Mean Squared Error (MSE) logic.
3. **Weight Update**: 
   $$\Delta w = \eta \cdot \delta \cdot \text{input}$$
   *(Where $\eta$ is the learning rate and $\delta$ is the calculated local gradient)*

## Getting Started

### Prerequisites
You only need a C compiler (like `gcc` or `clang`).

### Compilation
Clone the repository and compile using GCC:
```bash
gcc NN.c -o NN -lm
```
(NOTE : The `-lm` flag is required to link the math lib for `exp()` function.)

## Running the program
```bash
./NN
```

## Expected Output
The network starts with random weights and high error. Over 10,000 epochs, you will see the Predicted Output converge toward the Target:
<img width="675" height="438" alt="image" src="https://github.com/user-attachments/assets/d8d73335-dd80-4de9-b271-a89674571124" />


## Things learned
    
    * Vanishing Gradients: How weight initialization can affect the start of training.
    * Memory Management: Handling multi-dimensional arrays in C to represent weight matrices.
    * Gradient Descent: The importance of shuffling training sets to avoid local minima.
