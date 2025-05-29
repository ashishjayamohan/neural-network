use neural_network::{NeuralNetwork, ReLU, Softmax};
use std::sync::Arc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand::distributions::Standard;

fn main() {
    println!("\n=== Multi-Class Classification (3 Classes) ===");
    
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    
    let samples_per_class = 50;
    let num_classes = 3;
    let total_samples = samples_per_class * num_classes;
    
    let mut inputs = Vec::with_capacity(total_samples);
    let mut targets = Vec::with_capacity(total_samples);
    
    let centers = [
        [0.2, 0.2],
        [0.8, 0.2],
        [0.5, 0.8]
    ];
    
    for c in 0..num_classes {
        for _ in 0..samples_per_class {
            let x = centers[c][0] + rng.gen::<f64>() * 0.1 - 0.05;
            let y = centers[c][1] + rng.gen::<f64>() * 0.1 - 0.05;
            
            let mut target = vec![0.0; num_classes];
            target[c] = 1.0;
            
            inputs.push(vec![x, y]);
            targets.push(target);
        }
    }
    
    let mut nn = NeuralNetwork::new(0.1);
    nn.add_input_layer(2, 16, Arc::new(ReLU)).unwrap();
    nn.add_layer(8, Arc::new(ReLU)).unwrap();
    nn.add_layer(num_classes, Arc::new(Softmax)).unwrap();
    
    println!("Training Multi-Class Classification network...");
    nn.fit(&inputs, &targets, 300, true).unwrap();
    
    println!("\nMulti-Class Classification Test Results:");
    let mut correct = 0;
    let num_test_points = 15;
    
    for _ in 0..num_test_points {
        let idx = rng.gen_range(0..total_samples);
        let input = &inputs[idx];
        let output = nn.predict(input).unwrap();
        
        let mut predicted_class = 0;
        for j in 1..num_classes {
            if output[j] > output[predicted_class] {
                predicted_class = j;
            }
        }
        
        let mut actual_class = 0;
        for j in 1..num_classes {
            if targets[idx][j] > targets[idx][actual_class] {
                actual_class = j;
            }
        }
        
        if predicted_class == actual_class {
            correct += 1;
        }
        
        println!("Point ({:.2}, {:.2}): Predicted Class {}, Actual Class {}",
            input[0], input[1], predicted_class, actual_class);
    }
    
    let accuracy = (correct as f64 / num_test_points as f64) * 100.0;
    println!("Test Accuracy: {:.2}% ({}/{} correct)", 
             accuracy, correct, num_test_points);
}