use neural_network::{NeuralNetwork, ReLU, Sigmoid};
use std::sync::Arc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() {
    println!("\n=== Function Approximation (Sine Wave) ===");
    
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    
    let num_samples = 100;
    let mut inputs = Vec::with_capacity(num_samples);
    let mut targets = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let x = (i as f64 / num_samples as f64) * 2.0 * std::f64::consts::PI;
        inputs.push(vec![x / (2.0 * std::f64::consts::PI)]);
        targets.push(vec![(x.sin() + 1.0) / 2.0]);
    }
    
    let mut nn = NeuralNetwork::new(0.05);
    nn.add_input_layer(1, 16, Arc::new(ReLU)).unwrap();
    nn.add_layer(16, Arc::new(ReLU)).unwrap();
    nn.add_layer(1, Arc::new(Sigmoid)).unwrap();
    
    println!("Training Sine Wave Approximation network...");
    nn.fit(&inputs, &targets, 5000, true).unwrap();
    
    println!("\nSine Wave Approximation Test Results:");
    let mut mse = 0.0;
    let num_test_points = 10;
    
    println!("x\tActual\tPredicted\tError");
    println!("---\t------\t---------\t-----");
    
    for _ in 0..num_test_points {
        let idx = rng.gen_range(0..num_samples);
        let input = &inputs[idx];
        let output = nn.predict(input).unwrap();
        
        let actual = targets[idx][0];
        let predicted = output[0];
        let error = (actual - predicted).abs();
        mse += error * error;
        
        let original_x = input[0] * 2.0 * std::f64::consts::PI;
        let original_actual = actual * 2.0 - 1.0;
        let original_predicted = predicted * 2.0 - 1.0;
        
        println!("{:.2}\t{:.4}\t{:.4}\t{:.4}", 
            original_x, original_actual, original_predicted, (original_actual - original_predicted).abs());
    }
    
    mse /= num_test_points as f64;
    println!("\nMean Squared Error: {:.6}", mse);
}
