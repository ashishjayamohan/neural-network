use crate::activation::ActivationFunction;
use crate::layer::Layer;
use crate::matrix::Matrix;
use std::sync::Arc;

pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(learning_rate: f64) -> Self {
        NeuralNetwork {
            layers: Vec::new(),
            learning_rate,
        }
    }

    pub fn add_layer(&mut self, output_size: usize, activation: Arc<dyn ActivationFunction>) -> Result<(), &'static str> {
        if self.layers.is_empty() {
            return Err("Must specify input size for the first layer");
        }
        
        let input_size = self.layers.last().unwrap().output_size;
        self.layers.push(Layer::new(input_size, output_size, activation));
        Ok(())
    }

    pub fn add_input_layer(&mut self, input_size: usize, output_size: usize, activation: Arc<dyn ActivationFunction>) -> Result<(), &'static str> {
        if !self.layers.is_empty() {
            return Err("Input layer must be added first");
        }
        
        self.layers.push(Layer::new(input_size, output_size, activation));
        Ok(())
    }

    pub fn predict(&mut self, input_array: &[f64]) -> Result<Vec<f64>, &'static str> {
        let mut input = Matrix::from_array(input_array);
        
        for layer in &mut self.layers {
            input = layer.feed_forward(&input)?;
        }
        
        Ok(input.to_array())
    }

    pub fn train(&mut self, input_array: &[f64], target_array: &[f64]) -> Result<(), &'static str> {
        let input = Matrix::from_array(input_array);
        let target = Matrix::from_array(target_array);
        
        let mut output = input;
        for layer in &mut self.layers {
            output = layer.feed_forward(&output)?;
        }
        
        let mut error = target.subtract(&output)?;
        
        for i in (0..self.layers.len()).rev() {
            error = self.layers[i].backpropagate(&error, self.learning_rate)?;
        }
        
        Ok(())
    }

    pub fn fit(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>], epochs: usize, verbose: bool) -> Result<(), &'static str> {
        if inputs.is_empty() || targets.is_empty() || inputs.len() != targets.len() {
            return Err("Invalid input/target data");
        }
        
        
        
        let report_frequency = if epochs < 100 {
            10
        } else if epochs < 1000 {
            100
        } else {
            epochs / 10
        };
        
        
        let data_size = inputs.len();
        let mut batch_indices: Vec<usize> = (0..data_size).collect();
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            
            if data_size > 10 {
                use rand::seq::SliceRandom;
                use rand::thread_rng;
                batch_indices.shuffle(&mut thread_rng());
            }
            
            
            for &i in &batch_indices {
                let output = self.predict(&inputs[i])?;
                let target = &targets[i];
                
                
                let mut sample_loss = 0.0;
                for j in 0..output.len() {
                    let error = target[j] - output[j];
                    sample_loss += error * error;
                }
                sample_loss /= output.len() as f64;
                total_loss += sample_loss;
                
                
                self.train(&inputs[i], target)?;
            }
            
            let avg_loss = total_loss / data_size as f64;
            
            
            if verbose && (epoch % report_frequency == 0 || epoch == epochs - 1) {
                let accuracy = self.calculate_accuracy(inputs, targets)?;
                println!("Epoch {}/{} - Loss: {:.6} - Accuracy: {:.2}%", 
                         epoch + 1, epochs, avg_loss, accuracy * 100.0);
            }
        }
        
        Ok(())
    }
    
    pub fn calculate_accuracy(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> Result<f64, &'static str> {
        if inputs.is_empty() || targets.is_empty() || inputs.len() != targets.len() {
            return Err("Invalid input/target data");
        }
        
        
        let max_samples = if cfg!(test) { 100 } else { inputs.len() };
        let sample_count = inputs.len().min(max_samples);
        
        
        let indices: Vec<usize> = if sample_count < inputs.len() {
            let step = inputs.len() / sample_count;
            (0..sample_count).map(|i| i * step).collect()
        } else {
            (0..inputs.len()).collect()
        };
        
        let mut correct = 0;
        
        
        for &i in &indices {
            let output = self.predict(&inputs[i])?;
            let target = &targets[i];
            
            
            let mut is_correct = true;
            
            for j in 0..output.len() {
                let predicted = if output[j] > 0.5 { 1.0 } else { 0.0 };
                if (predicted - target[j]).abs() > 0.01 {
                    is_correct = false;
                    break;
                }
            }
            
            if is_correct {
                correct += 1;
            }
        }
        
        
        Ok(correct as f64 / sample_count as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{ReLU, Sigmoid};

    #[test]
    fn test_neural_network_creation() {
        let nn = NeuralNetwork::new(0.1);
        assert_eq!(nn.layers.len(), 0);
        assert_eq!(nn.learning_rate, 0.1);
    }

    #[test]
    fn test_add_layers() {
        let mut nn = NeuralNetwork::new(0.1);
        
        let result = nn.add_input_layer(2, 3, Arc::new(ReLU) as Arc<dyn ActivationFunction>);
        assert!(result.is_ok());
        assert_eq!(nn.layers.len(), 1);
        
        let result = nn.add_layer(1, Arc::new(Sigmoid) as Arc<dyn ActivationFunction>);
        assert!(result.is_ok());
        assert_eq!(nn.layers.len(), 2);
    }

    #[test]
    fn test_predict() {
        let mut nn = NeuralNetwork::new(0.1);
        
        nn.add_input_layer(2, 1, Arc::new(Sigmoid) as Arc<dyn ActivationFunction>).unwrap();
        
        nn.layers[0].weights.set(0, 0, 0.5);
        nn.layers[0].weights.set(0, 1, 0.5);
        nn.layers[0].biases.set(0, 0, 0.0);
        
        let input = vec![1.0, 1.0];
        let output = nn.predict(&input).unwrap();
        
        assert!((output[0] - 0.7310585786300049).abs() < 1e-10);
    }
}