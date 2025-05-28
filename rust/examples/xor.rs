use neural_network::{NeuralNetwork, ReLU, Sigmoid};
use std::sync::Arc;

fn main() {
    println!("\n=== XOR Problem ===");

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    let mut nn = NeuralNetwork::new(0.1);
    nn.add_input_layer(2, 3, Arc::new(ReLU)).unwrap();
    nn.add_layer(1, Arc::new(Sigmoid)).unwrap();

    println!("Training XOR network...");
    nn.fit(&inputs, &targets, 2000, true).unwrap();

    println!("\nDetailed XOR Test Results:");
    for i in 0..inputs.len() {
        let input = &inputs[i];
        let output = nn.predict(input).unwrap();
        let raw_output = output[0];
        let predicted = if raw_output > 0.5 { 1.0 } else { 0.0 };

        println!("Input: [{}, {}] -> Raw Output: {:.6} -> Predicted: {:.0} (Expected: {:.0})",
            input[0] as i32, input[1] as i32, raw_output, predicted, targets[i][0]);
    }

    let accuracy = nn.calculate_accuracy(&inputs, &targets).unwrap() * 100.0;
    println!("XOR Accuracy: {:.2}%", accuracy);
}
