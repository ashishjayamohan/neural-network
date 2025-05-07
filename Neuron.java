public class Neuron {
  private double[] weights;
  private double bias;
  private ActivationFunction activation;

  public Neuron(int inputSize, ActivationFunction activation) {
    this.weights = new double[inputSize];
    this.bias = Math.random() * 2 - 1;
    this.activation = activation;

    for (int i = 0; i < inputSize; i++) {
      weights[i] = Math.random() * 2 - 1;
    }
  }

  public double feedForward(double[] inputs) {
    if (inputs.length != weights.length) {
      throw new IllegalArgumentException("Input size must match weight size.");
    }

    double sum = 0.0;
    for (int i = 0; i < inputs.length; i++) {
      sum += inputs[i] * weights[i];
    }
    sum += bias;

    return activation.activate(sum); // ✅ changed apply -> activate
  }

  public double[] getWeights() {
    return weights;
  }

  public double getBias() {
    return bias;
  }

  public void setWeights(double[] weights) {
    if (weights.length != this.weights.length) {
      throw new IllegalArgumentException("Weights array must match input size.");
    }
    this.weights = weights;
  }

  public void setBias(double bias) {
    this.bias = bias;
  }
}
