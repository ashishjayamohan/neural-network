public class SigmoidActivation extends ActivationFunction {
  @Override
  public double activate(double x) {
    return 1.0 / (1.0 + Math.exp(-x));
  }

  @Override
  public double derivative(double y) {
    return y * (1 - y);
  }
}
