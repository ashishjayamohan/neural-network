public class ReLUActivation extends ActivationFunction {
  @Override
  public double activate(double x) {
    return Math.max(0, x);
  }

  @Override
  public double derivative(double y) {
    return y > 0 ? 1 : 0;
  }
}
