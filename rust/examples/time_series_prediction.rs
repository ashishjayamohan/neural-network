use neural_network::{NeuralNetwork, ReLU, Sigmoid};
use std::sync::Arc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() {
    println!("\n=== Time Series Prediction (Stock Price Simulation) ===");
    
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    
    let data_points = 200;
    let mut time_series_data = Vec::with_capacity(data_points);
    
    let base_value = 100.0;
    
    for i in 0..data_points {
        let trend = 0.05 * i as f64;
        
        let seasonality = 5.0 * (i as f64 * 2.0 * std::f64::consts::PI / 20.0).sin();
        
        let noise = rng.gen::<f64>() * 4.0 - 2.0; 
        
        time_series_data.push(base_value + trend + seasonality + noise);
    }
    
    let window_size = 5;
    let num_samples = data_points - window_size;
    
    let mut inputs = Vec::with_capacity(num_samples);
    let mut targets = Vec::with_capacity(num_samples);
    
    let min_value = *time_series_data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_value = *time_series_data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let range = max_value - min_value;
    
    for i in 0..num_samples {
        let mut input = Vec::with_capacity(window_size);
        for j in 0..window_size {
            input.push((time_series_data[i + j] - min_value) / range);
        }
        
        let target = vec![(time_series_data[i + window_size] - min_value) / range];
        
        inputs.push(input);
        targets.push(target);
    }
    
    let train_size = (num_samples as f64 * 0.8) as usize;
    let test_size = num_samples - train_size;
    
    let train_inputs = inputs[0..train_size].to_vec();
    let train_targets = targets[0..train_size].to_vec();
    let test_inputs = inputs[train_size..].to_vec();
    let test_targets = targets[train_size..].to_vec();
    
    let mut nn = NeuralNetwork::new(0.01);
    nn.add_input_layer(window_size, 16, Arc::new(ReLU)).unwrap();
    nn.add_layer(8, Arc::new(ReLU)).unwrap();
    nn.add_layer(1, Arc::new(Sigmoid)).unwrap();
    
    println!("Training Time Series Prediction network...");
    nn.fit(&train_inputs, &train_targets, 2000, true).unwrap();
    
    println!("\nTime Series Prediction Test Results:");
    let mut total_error = 0.0;
    
    println!("Day\tActual\tPredicted\tError");
    println!("---\t------\t---------\t-----");
    
    for i in 0..test_size {
        let input = &test_inputs[i];
        let output = nn.predict(input).unwrap();
        
        let actual_normalized = test_targets[i][0];
        let predicted_normalized = output[0];
        
        let actual = actual_normalized * range + min_value;
        let predicted = predicted_normalized * range + min_value;
        
        let error = (actual - predicted).abs();
        total_error += error;
        
        if i % 5 == 0 {
            println!("{}\t{:.2}\t{:.2}\t{:.2}", 
                i + train_size + window_size, actual, predicted, error);
        }
    }
    
    let mae = total_error / test_size as f64;
    println!("\nMean Absolute Error: {:.2}", mae);
    println!("Mean Absolute Percentage Error: {:.2}%", (mae / ((max_value + min_value) / 2.0)) * 100.0);
    
    println!("\nFuture Predictions (next 10 days):");
    println!("Day\tPredicted Value");
    println!("---\t--------------");
    
    let mut prediction_window = Vec::with_capacity(window_size);
    for i in 0..window_size {
        prediction_window.push((time_series_data[data_points - window_size + i] - min_value) / range);
    }
    
    for i in 0..10 {
        let output = nn.predict(&prediction_window).unwrap();
        let predicted_value = output[0] * range + min_value;
        
        println!("{}\t{:.2}", data_points + i, predicted_value);
        
        for j in 0..(window_size - 1) {
            prediction_window[j] = prediction_window[j + 1];
        }
        prediction_window[window_size - 1] = output[0];
    }
}
