public class SoftmaxActivation extends ActivationFunction {
    @Override
    public double activate(double x) {

        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public double derivative(double y) {

        return y * (1 - y);
    }
    
    @Override
    public double[] activate(double[] input) {
        double[] output = new double[input.length];
        

        double maxVal = input[0];
        for (int i = 1; i < input.length; i++) {
            if (input[i] > maxVal) {
                maxVal = input[i];
            }
        }
        

        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i] - maxVal);
            sum += output[i];
        }
        

        if (sum < 1e-10) {
            double uniformProb = 1.0 / input.length;
            for (int i = 0; i < output.length; i++) {
                output[i] = uniformProb;
            }
        } else {
            for (int i = 0; i < output.length; i++) {
                output[i] /= sum;
            }
        }
        
        return output;
    }
}
