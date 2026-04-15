# Lab Assignment 1

## Neural Network from Scratch

This assignment implements a simple feedforward neural network from scratch using NumPy to demonstrate the fundamental concepts of forward propagation and backpropagation.

### Overview

The implementation includes:

- **Sigmoid Activation Function**: Converts input values to range (0, 1)
- **Forward Propagation**: Computes network output from inputs
- **Backpropagation**: Updates weights using gradient descent
- **Training Loop**: Iteratively improves network performance

### Network Architecture

- **Input Layer**: 2 neurons (x1, x2)
- **Hidden Layer**: 2 neurons (h1, h2)
- **Output Layer**: 1 neuron (y)

### Key Features

1. **Custom Weight Initialization**: Uses specific weight values from a predefined example

   - Input to Hidden weights: `[[-0.5, 0.4], [0.9, 0.1]]`
   - Hidden to Output weights: `[[0.2], [-0.6]]`
2. **Activation Functions**:

   - Sigmoid function for non-linearity
   - Sigmoid derivative for gradient computation
3. **Training**:

   - Learning rate: 0.1
   - Mean Squared Error (MSE) loss tracking
   - Single example training demonstration

### Files

- `lab_assisgnment_1_colab.ipynb`: Main notebook with complete implementation

### Usage

The notebook demonstrates:

1. Network initialization with specific weights
2. Forward pass computation
3. Single backpropagation step
4. Weight updates using gradient descent

### Example Input/Output

**Input**: `[0, 1]`
**Target**: `[0]`
**Learning Rate**: `0.1`

The notebook shows the network output before and after one training step to illustrate the learning process.

### Implementation Details

- Pure NumPy implementation (no deep learning frameworks)
- Manual weight initialization for educational purposes
- Step-by-step forward and backward propagation
- Loss tracking every 1000 epochs during extended training
