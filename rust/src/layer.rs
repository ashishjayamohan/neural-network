use crate::activation::ActivationFunction;
use crate::matrix::Matrix;
use std::sync::Arc;

pub struct Layer {
    pub output_size: usize,
    pub weights: Matrix,
    pub biases: Matrix,
    activation: Arc<dyn ActivationFunction>,
    last_input: Option<Matrix>,
    last_activation: Option<Matrix>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: Arc<dyn ActivationFunction>) -> Self {
        let mut weights = Matrix::new(output_size, input_size);
        let mut biases = Matrix::new(output_size, 1);
        weights.randomize();
        biases.randomize();

        Layer {
            output_size,
            weights,
            biases,
            activation,
            last_input: None,
            last_activation: None,
        }
    }

    pub fn feed_forward(&mut self, input: &Matrix) -> Result<Matrix, &'static str> {
        
        self.last_input = Some(input.clone());
        
        
        
        let z = Matrix::dot(&self.weights, input)?.add(&self.biases)?;
        
        
        
        let activation_output = z.map(|x| self.activation.activate(x));
        
        
        self.last_activation = Some(activation_output.clone());
        
        Ok(activation_output)
    }

    pub fn backpropagate(&mut self, output_error: &Matrix, learning_rate: f64) -> Result<Matrix, &'static str> {
        let last_input = self.last_input.as_ref().ok_or("No input stored for backpropagation")?;
        let last_activation = self.last_activation.as_ref().ok_or("No activation stored for backpropagation")?;
        
        
        let activation_derivative = last_activation.map(|x| self.activation.derivative(x));
        
        
        let delta = Matrix::hadamard(output_error, &activation_derivative)?;
        
        
        let input_transpose = Matrix::transpose(last_input);
        let weight_gradient = Matrix::dot(&delta, &input_transpose)?;
        
        
        let weight_delta = weight_gradient.multiply(learning_rate);
        let bias_delta = delta.multiply(learning_rate);
        
        
        let w_rows = self.weights.rows;
        let w_cols = self.weights.cols;
        
        
        for i in 0..w_rows {
            for j in 0..w_cols {
                let idx = i * w_cols + j;
                self.weights.data[idx] += weight_delta.data[idx];
            }
        }
        
        
        for i in 0..self.biases.rows {
            self.biases.data[i] += bias_delta.data[i];
        }
        
        
        let weights_transpose = Matrix::transpose(&self.weights);
        Matrix::dot(&weights_transpose, &delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{ReLU, Sigmoid};

    #[test]
    fn test_layer_creation() {
        let activation = Arc::new(ReLU) as Arc<dyn ActivationFunction>;
        let layer = Layer::new(2, 3, activation);
        assert_eq!(layer.output_size, 3);
    }

    #[test]
    fn test_feed_forward() {
        let activation = Arc::new(Sigmoid) as Arc<dyn ActivationFunction>;
        let mut layer = Layer::new(2, 1, activation);
        
        layer.weights.set(0, 0, 0.5);
        layer.weights.set(0, 1, 0.5);
        layer.biases.set(0, 0, 0.0);
        
        let input = Matrix::from_array(&[1.0, 1.0]);
        let output = layer.feed_forward(&input).unwrap();
        
        assert!((output.get(0, 0) - 0.7310585786300049).abs() < 1e-10);
    }
}
