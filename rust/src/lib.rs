pub mod activation;
pub mod layer;
pub mod matrix;
pub mod neural_network;

pub use activation::{ActivationFunction, ReLU, Sigmoid, Softmax};
pub use layer::Layer;
pub use matrix::Matrix;
pub use neural_network::NeuralNetwork;
