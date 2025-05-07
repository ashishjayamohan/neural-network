import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
  private final List<Layer> layers = new ArrayList<>();
  private double learningRate;

  public NeuralNetwork(double learningRate) {
    this.learningRate = learningRate;
  }

  public void addLayer(int outputSize, ActivationFunction activation) {
    int inputSize = layers.isEmpty() ? -1 : layers.get(layers.size() - 1).outputSize;
    if (inputSize == -1) {
      throw new IllegalStateException("Must specify input size for the first layer");
    }
    layers.add(new Layer(inputSize, outputSize, activation));
  }

  public void addInputLayer(int inputSize, int outputSize, ActivationFunction activation) {
    if (!layers.isEmpty()) {
      throw new IllegalStateException("Input layer must be added first");
    }
    layers.add(new Layer(inputSize, outputSize, activation));
  }

  public double[] predict(double[] inputArray) {
    Matrix input = Matrix.fromArray(inputArray);
    Matrix output = input;
    for (Layer layer : layers) {
      output = layer.feedForward(output);
    }
    return output.toArray();
  }

  public void train(double[] inputArray, double[] targetArray) {
    Matrix input = Matrix.fromArray(inputArray);
    Matrix target = Matrix.fromArray(targetArray);

    Matrix output = input;
    for (Layer layer : layers) {
      output = layer.feedForward(output);
    }

    Matrix error = target.subtract(output);
    for (int i = layers.size() - 1; i >= 0; i--) {
      error = layers.get(i).backpropagate(error, learningRate);
    }
  }


  public void fit(double[][] inputs, double[][] targets, int epochs, boolean verbose) {
    for (int e = 0; e < epochs; e++) {
      double totalLoss = 0;
      
      for (int i = 0; i < inputs.length; i++) {
        double[] output = predict(inputs[i]);
        double[] target = targets[i];
        

        double sampleLoss = 0;
        for (int j = 0; j < output.length; j++) {
          double error = target[j] - output[j];
          sampleLoss += error * error;
        }
        sampleLoss /= output.length;
        totalLoss += sampleLoss;
        

        train(inputs[i], targets[i]);
      }
      

      double avgLoss = totalLoss / inputs.length;
      

      if (verbose && (e % 100 == 0 || e == epochs - 1)) {
        double accuracy = calculateAccuracy(inputs, targets);
        System.out.printf("Epoch %d/%d - Loss: %.6f - Accuracy: %.2f%%\n", 
                         e+1, epochs, avgLoss, accuracy * 100);
      }
    }
  }
  
  public void fit(double[][] inputs, double[][] targets, int epochs) {
    fit(inputs, targets, epochs, false);
  }
  
  public double calculateAccuracy(double[][] inputs, double[][] targets) {
    int correct = 0;
    
    for (int i = 0; i < inputs.length; i++) {
      double[] output = predict(inputs[i]);
      boolean isCorrect = true;
      
      for (int j = 0; j < output.length; j++) {

        double predicted = output[j] > 0.5 ? 1.0 : 0.0;
        if (Math.abs(predicted - targets[i][j]) > 0.01) {
          isCorrect = false;
          break;
        }
      }
      
      if (isCorrect) {
        correct++;
      }
    }
    
    return (double) correct / inputs.length;
  }
}
