pub trait ActivationFunction: Send + Sync {
    fn activate(&self, x: f64) -> f64;
    fn derivative(&self, y: f64) -> f64;
    
    fn activate_vec(&self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| self.activate(x)).collect()
    }
    
    fn derivative_vec(&self, output: &[f64]) -> Vec<f64> {
        output.iter().map(|&y| self.derivative(y)).collect()
    }
}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn activate(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, y: f64) -> f64 {
        y * (1.0 - y)
    }
}

pub struct ReLU;

impl ActivationFunction for ReLU {
    fn activate(&self, x: f64) -> f64 {
        x.max(0.0)
    }

    fn derivative(&self, y: f64) -> f64 {
        if y > 0.0 { 1.0 } else { 0.0 }
    }
}

pub struct Softmax;

impl ActivationFunction for Softmax {
    fn activate(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, y: f64) -> f64 {
        y * (1.0 - y)
    }
    
    fn activate_vec(&self, input: &[f64]) -> Vec<f64> {
        if input.is_empty() {
            return Vec::new();
        }
        
        let max_val = input.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let mut output: Vec<f64> = input.iter()
            .map(|&x| (x - max_val).exp())
            .collect();
        
        let sum: f64 = output.iter().sum();
        
        if sum < 1e-10 {
            let uniform_prob = 1.0 / input.len() as f64;
            for val in &mut output {
                *val = uniform_prob;
            }
        } else {
            for val in &mut output {
                *val /= sum;
            }
        }
        
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid;
        assert_eq!(sigmoid.activate(0.0), 0.5);
        assert!((sigmoid.activate(2.0) - 0.8807970779778823).abs() < 1e-10);
        assert!((sigmoid.derivative(0.5) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_relu() {
        let relu = ReLU;
        assert_eq!(relu.activate(1.0), 1.0);
        assert_eq!(relu.activate(-1.0), 0.0);
        assert_eq!(relu.derivative(1.0), 1.0);
        assert_eq!(relu.derivative(0.0), 0.0);
    }

    #[test]
    fn test_softmax() {
        let softmax = Softmax;
        let input = vec![2.0, 1.0, 0.1];
        let output = softmax.activate_vec(&input);
        
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        
        assert!(output[0] > output[1]);
        assert!(output[1] > output[2]);
    }
}
