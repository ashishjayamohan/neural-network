public abstract class ActivationFunction {
  public abstract double activate(double x);
  public abstract double derivative(double y);

  public double[] activate(double[] input) {
    double[] output = new double[input.length];
    for (int i = 0; i < input.length; i++) {
      output[i] = activate(input[i]);
    }
    return output;
  }

  public double[] derivative(double[] output) {
    double[] result = new double[output.length];
    for (int i = 0; i < output.length; i++) {
      result[i] = derivative(output[i]);
    }
    return result;
  }
}
