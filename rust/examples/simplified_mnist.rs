use neural_network::{NeuralNetwork, ReLU, Softmax};
use std::sync::Arc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() {
    println!("\n=== Simplified MNIST Digit Recognition (0-3) ===");
    
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    
    let digit_patterns = [
        [
            [0, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 0]
        ],
        [
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0]
        ],
        [
            [0, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0]
        ],
        [
            [0, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 0]
        ]
    ];
    
    let num_digits = digit_patterns.len();
    let pixels_per_digit = 8 * 8;
    let num_samples = num_digits * 5;
    
    let mut inputs = Vec::with_capacity(num_samples);
    let mut targets = Vec::with_capacity(num_samples);
    
    for digit in 0..num_digits {
        for variation in 0..5 {
            let mut input = Vec::with_capacity(pixels_per_digit);
            
            for i in 0..8 {
                for j in 0..8 {
                    let noise = rng.gen::<f64>();
                    if noise < 0.1 && variation > 0 {
                        input.push(1.0 - digit_patterns[digit][i][j] as f64);
                    } else {
                        input.push(digit_patterns[digit][i][j] as f64);
                    }
                }
            }
            
            let mut target = vec![0.0; num_digits];
            target[digit] = 1.0;
            
            inputs.push(input);
            targets.push(target);
        }
    }
    
    let mut nn = NeuralNetwork::new(0.05);
    nn.add_input_layer(pixels_per_digit, 64, Arc::new(ReLU)).unwrap();
    nn.add_layer(32, Arc::new(ReLU)).unwrap();
    nn.add_layer(num_digits, Arc::new(Softmax)).unwrap();
    
    println!("Training Simplified MNIST network...");
    nn.fit(&inputs, &targets, 2000, true).unwrap();
    
    println!("\nSimplified MNIST Test Results:");
    
    for digit in 0..num_digits {
        let mut input = Vec::with_capacity(pixels_per_digit);
        for i in 0..8 {
            for j in 0..8 {
                input.push(digit_patterns[digit][i][j] as f64);
            }
        }
        
        let output = nn.predict(&input).unwrap();
        
        let mut predicted_digit = 0;
        for d in 1..num_digits {
            if output[d] > output[predicted_digit] {
                predicted_digit = d;
            }
        }
        
        println!("Digit {}: Predicted as {}, Confidence: {:.2}%",
            digit, predicted_digit, output[predicted_digit] * 100.0);
        
        print!("Class probabilities: ");
        for d in 0..num_digits {
            print!("[{}]: {:.2}% ", d, output[d] * 100.0);
        }
        println!();
        
        println!("Pattern:");
        for i in 0..8 {
            for j in 0..8 {
                print!("{}", if digit_patterns[digit][i][j] == 1 { "█" } else { " " });
            }
            println!();
        }
        println!();
    }
}
