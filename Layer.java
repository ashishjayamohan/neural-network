public class Layer {
  final int outputSize;
  private Matrix weights;
  private Matrix biases;
  private final ActivationFunction activation;

  private Matrix lastInput;

  private Matrix lastActivation;

  public Layer(int inputSize, int outputSize, ActivationFunction activation) {
    this.outputSize = outputSize;
    this.activation = activation;

    this.weights = new Matrix(outputSize, inputSize);
    this.biases = new Matrix(outputSize, 1);
    weights.randomize();
    biases.randomize();
  }

  public Matrix feedForward(Matrix input) {
    this.lastInput = input;
    Matrix z = Matrix.dot(weights, input).add(biases);
    Matrix activation_output = z.map(activation::activate);
    this.lastActivation = activation_output;
    return activation_output;
  }

  public Matrix backpropagate(Matrix outputError, double learningRate) {

    Matrix activationDerivative = lastActivation.map(activation::derivative);
    Matrix delta = Matrix.hadamard(outputError, activationDerivative);

    Matrix inputTranspose = Matrix.transpose(lastInput);
    Matrix weightGradient = Matrix.dot(delta, inputTranspose);


    Matrix weightDelta = weightGradient.multiply(learningRate);
    Matrix biasDelta = delta.multiply(learningRate);
    

    for (int i = 0; i < weights.rows; i++) {
      for (int j = 0; j < weights.cols; j++) {
        weights.set(i, j, weights.get(i, j) + weightDelta.get(i, j));
      }
    }
    
    for (int i = 0; i < biases.rows; i++) {
      biases.set(i, 0, biases.get(i, 0) + biasDelta.get(i, 0));
    }

    Matrix weightsTranspose = Matrix.transpose(weights);
    return Matrix.dot(weightsTranspose, delta);
  }
}
