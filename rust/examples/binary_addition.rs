use neural_network::{NeuralNetwork, ReLU, Sigmoid};
use std::sync::Arc;

fn main() {
    println!("\n=== Binary Addition Problem ===");

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let targets = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
    ];

    let mut nn = NeuralNetwork::new(0.1);
    nn.add_input_layer(2, 8, Arc::new(ReLU)).unwrap();
    nn.add_layer(2, Arc::new(Sigmoid)).unwrap();

    println!("Training Binary Addition network...");
    nn.fit(&inputs, &targets, 2000, true).unwrap();

    println!("\nDetailed Binary Addition Test Results:");
    for i in 0..inputs.len() {
        let input = &inputs[i];
        let output = nn.predict(input).unwrap();
        let mut predicted = vec![0.0; output.len()];
        
        for j in 0..output.len() {
            predicted[j] = if output[j] > 0.5 { 1.0 } else { 0.0 };
        }

        println!("Input: [{}, {}] -> Raw Output: [{:.6}, {:.6}] -> Predicted: [{:.0}, {:.0}] (Expected: [{:.0}, {:.0}])",
            input[0] as i32, input[1] as i32, 
            output[0], output[1],
            predicted[0], predicted[1], 
            targets[i][0], targets[i][1]);
    }

    let accuracy = nn.calculate_accuracy(&inputs, &targets).unwrap() * 100.0;
    println!("Binary Addition Accuracy: {:.2}%", accuracy);
}
