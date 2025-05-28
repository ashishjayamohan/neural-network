# Neural Network in Rust

This is a Rust implementation of the neural network originally written in Java. The implementation includes a matrix library, activation functions, and a neural network with backpropagation.

## Features

- Matrix operations for neural network computations
- Activation functions: Sigmoid, ReLU, and Softmax
- Neural network with configurable layers and activation functions
- Example implementations of common problems:
  - XOR problem
  - Binary addition
  - Function approximation (sine wave)
  - Multi-class classification
  - Simplified MNIST digit recognition
  - Time series prediction

## Requirements

- Rust (edition 2021 or later)
- Cargo (Rust's package manager)

## How to Run

### Building the Library

```bash
cd rust
cargo build --release
```

### Running the Examples

You can run any of the examples using:

```bash
# Run the XOR example
cargo run --example xor

# Run the binary addition example
cargo run --example binary_addition

# Run the function approximation example
cargo run --example function_approximation

# Run the multi-class classification example
cargo run --example multi_class_classification

# Run the simplified MNIST example
cargo run --example simplified_mnist

# Run the time series prediction example
cargo run --example time_series_prediction
```

## Implementation Details

The neural network implementation consists of the following components:

1. **Matrix**: A matrix implementation with operations like dot product, addition, subtraction, and element-wise operations.

2. **Activation Functions**:
   - Sigmoid: `f(x) = 1 / (1 + e^(-x))`
   - ReLU: `f(x) = max(0, x)`
   - Softmax: For multi-class classification

3. **Layer**: Represents a layer in the neural network with weights, biases, and an activation function.

4. **Neural Network**: Combines multiple layers and implements the training and prediction algorithms.

## Testing

The implementation includes unit tests for each component. You can run the tests with:

```bash
cargo test
```

## Performance

The Rust implementation should provide better performance than the Java version due to Rust's efficiency and lack of garbage collection overhead.

## License

This project is open source and available under the same license as the original Java implementation.
