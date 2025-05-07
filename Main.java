import java.util.Random;

public class Main {
    private static final Random random = new Random(42);
    
    public static void main(String[] args) {

        testXOR();
        testBinaryAddition();
        

        testFunctionApproximation();
        testMultiClassClassification();
        testSimplifiedMNIST();
        testTimeSeriesPrediction();
    }
    
    public static void testXOR() {
        System.out.println("\n=== XOR Problem ===");

        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[][] targets = {
            {0},
            {1},
            {1},
            {0}
        };


        NeuralNetwork nn = new NeuralNetwork(0.1);  // Higher learning rate
        nn.addInputLayer(2, 3, new ReLUActivation());
        nn.addLayer(1, new SigmoidActivation());

        // Train the network with verbose output
        System.out.println("Training XOR network...");
        nn.fit(inputs, targets, 2000, true);

        // Evaluate results
        System.out.println("\nDetailed XOR Test Results:");
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double[] output = nn.predict(input);
            double rawOutput = output[0];
            double predicted = rawOutput > 0.5 ? 1.0 : 0.0;

            System.out.printf("Input: [%d, %d] -> Raw Output: %.6f -> Predicted: %.0f (Expected: %.0f)%n",
                (int) input[0], (int) input[1], rawOutput, predicted, targets[i][0]);
        }

        double accuracy = nn.calculateAccuracy(inputs, targets) * 100;
        System.out.printf("XOR Accuracy: %.2f%%\n", accuracy);
    }
    
    public static void testBinaryAddition() {
        System.out.println("\n=== Binary Addition Problem ===");

        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[][] targets = {
            {0, 0},
            {1, 0},
            {1, 0},
            {0, 1}
        };


        NeuralNetwork nn = new NeuralNetwork(0.1);
        nn.addInputLayer(2, 8, new ReLUActivation());
        nn.addLayer(2, new SigmoidActivation());

        // Train the network with verbose output
        System.out.println("Training Binary Addition network...");
        nn.fit(inputs, targets, 2000, true);

        // Evaluate results
        System.out.println("\nDetailed Binary Addition Test Results:");
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double[] output = nn.predict(input);
            double[] predicted = new double[output.length];
            
            for (int j = 0; j < output.length; j++) {
                predicted[j] = output[j] > 0.5 ? 1.0 : 0.0;
            }

            System.out.printf("Input: [%d, %d] -> Raw Output: [%.6f, %.6f] -> Predicted: [%.0f, %.0f] (Expected: [%.0f, %.0f])%n",
                (int) input[0], (int) input[1], 
                output[0], output[1],
                predicted[0], predicted[1], 
                targets[i][0], targets[i][1]);
        }

        double accuracy = nn.calculateAccuracy(inputs, targets) * 100;
        System.out.printf("Binary Addition Accuracy: %.2f%%\n", accuracy);
    }
    

    public static void testFunctionApproximation() {
        System.out.println("\n=== Function Approximation (Sine Wave) ===");
        

        int numSamples = 100;
        double[][] inputs = new double[numSamples][1];
        double[][] targets = new double[numSamples][1];
        
        for (int i = 0; i < numSamples; i++) {
            double x = (i / (double) numSamples) * 2 * Math.PI;
            inputs[i][0] = x / (2 * Math.PI);
            targets[i][0] = (Math.sin(x) + 1) / 2;
        }
        

        NeuralNetwork nn = new NeuralNetwork(0.05);
        nn.addInputLayer(1, 16, new ReLUActivation());
        nn.addLayer(16, new ReLUActivation());
        nn.addLayer(1, new SigmoidActivation());
        
        // Train the network
        System.out.println("Training Sine Wave Approximation network...");
        nn.fit(inputs, targets, 5000, true);
        

        System.out.println("\nSine Wave Approximation Test Results:");
        double mse = 0.0;
        int numTestPoints = 10;
        
        for (int i = 0; i < numTestPoints; i++) {
            int idx = random.nextInt(numSamples);
            double[] input = inputs[idx];
            double[] output = nn.predict(input);
            double actual = output[0];
            double expected = targets[idx][0];
            double error = expected - actual;
            mse += error * error;
            

            double originalX = input[0] * 2 * Math.PI;
            double originalActual = actual * 2 - 1;
            double originalExpected = expected * 2 - 1;
            
            System.out.printf("x = %.2f: sin(x) = %.4f, predicted = %.4f, error = %.4f%n",
                originalX, originalExpected, originalActual, Math.abs(originalExpected - originalActual));
        }
        
        mse /= numTestPoints;
        System.out.printf("Mean Squared Error: %.6f%n", mse);
    }
    

    public static void testMultiClassClassification() {
        System.out.println("\n=== Multi-Class Classification (3 Classes) ===");
        

        int samplesPerClass = 50;
        int numClasses = 3;
        int totalSamples = samplesPerClass * numClasses;
        
        double[][] inputs = new double[totalSamples][2];
        double[][] targets = new double[totalSamples][numClasses];
        

        double[][] centers = {
            {0.2, 0.2},
            {0.8, 0.2},
            {0.5, 0.8}
        };
        
        for (int c = 0; c < numClasses; c++) {
            for (int i = 0; i < samplesPerClass; i++) {
                int idx = c * samplesPerClass + i;
                

                inputs[idx][0] = centers[c][0] + random.nextGaussian() * 0.1;
                inputs[idx][1] = centers[c][1] + random.nextGaussian() * 0.1;
                

                for (int j = 0; j < numClasses; j++) {
                    targets[idx][j] = (j == c) ? 1.0 : 0.0;
                }
            }
        }
        

        NeuralNetwork nn = new NeuralNetwork(0.1);
        nn.addInputLayer(2, 16, new ReLUActivation());
        nn.addLayer(8, new ReLUActivation());
        nn.addLayer(numClasses, new SoftmaxActivation());
        
        // Train the network
        System.out.println("Training Multi-Class Classification network...");
        nn.fit(inputs, targets, 3000, true);
        

        System.out.println("\nMulti-Class Classification Test Results:");
        int correct = 0;
        int numTestPoints = 15;
        
        for (int i = 0; i < numTestPoints; i++) {
            int idx = random.nextInt(totalSamples);
            double[] input = inputs[idx];
            double[] output = nn.predict(input);
            

            int predictedClass = 0;
            for (int j = 1; j < numClasses; j++) {
                if (output[j] > output[predictedClass]) {
                    predictedClass = j;
                }
            }
            

            int actualClass = 0;
            for (int j = 1; j < numClasses; j++) {
                if (targets[idx][j] > targets[idx][actualClass]) {
                    actualClass = j;
                }
            }
            
            if (predictedClass == actualClass) {
                correct++;
            }
            
            System.out.printf("Point (%.2f, %.2f): Predicted Class %d, Actual Class %d%n",
                input[0], input[1], predictedClass, actualClass);
        }
        
        double accuracy = (double) correct / numTestPoints * 100;
        System.out.printf("Test Accuracy: %.2f%% (%d/%d correct)%n", 
                         accuracy, correct, numTestPoints);
    }
    

    public static void testSimplifiedMNIST() {
        System.out.println("\n=== Simplified MNIST Digit Recognition (0-3) ===");
        

        int[][][] digitPatterns = {

            {
                {0, 1, 1, 1, 1, 1, 0, 0},
                {1, 1, 0, 0, 0, 1, 1, 0},
                {1, 0, 0, 0, 0, 0, 1, 0},
                {1, 0, 0, 0, 0, 0, 1, 0},
                {1, 0, 0, 0, 0, 0, 1, 0},
                {1, 0, 0, 0, 0, 0, 1, 0},
                {1, 1, 0, 0, 0, 1, 1, 0},
                {0, 1, 1, 1, 1, 1, 0, 0}
            },

            {
                {0, 0, 0, 1, 1, 0, 0, 0},
                {0, 0, 1, 1, 1, 0, 0, 0},
                {0, 1, 0, 1, 1, 0, 0, 0},
                {0, 0, 0, 1, 1, 0, 0, 0},
                {0, 0, 0, 1, 1, 0, 0, 0},
                {0, 0, 0, 1, 1, 0, 0, 0},
                {0, 0, 0, 1, 1, 0, 0, 0},
                {0, 1, 1, 1, 1, 1, 1, 0}
            },

            {
                {0, 1, 1, 1, 1, 1, 0, 0},
                {1, 1, 0, 0, 0, 1, 1, 0},
                {0, 0, 0, 0, 0, 1, 1, 0},
                {0, 0, 0, 0, 1, 1, 0, 0},
                {0, 0, 0, 1, 1, 0, 0, 0},
                {0, 0, 1, 1, 0, 0, 0, 0},
                {0, 1, 1, 0, 0, 0, 0, 0},
                {1, 1, 1, 1, 1, 1, 1, 0}
            },

            {
                {0, 1, 1, 1, 1, 1, 0, 0},
                {1, 1, 0, 0, 0, 1, 1, 0},
                {0, 0, 0, 0, 0, 1, 1, 0},
                {0, 0, 0, 1, 1, 1, 0, 0},
                {0, 0, 0, 0, 0, 1, 1, 0},
                {0, 0, 0, 0, 0, 1, 1, 0},
                {1, 1, 0, 0, 0, 1, 1, 0},
                {0, 1, 1, 1, 1, 1, 0, 0}
            }
        };
        
        int numDigits = digitPatterns.length;
        int pixelsPerDigit = 8 * 8;
        int numSamples = numDigits * 5;
        
        double[][] inputs = new double[numSamples][pixelsPerDigit];
        double[][] targets = new double[numSamples][numDigits];
        

        for (int digit = 0; digit < numDigits; digit++) {
            for (int variation = 0; variation < 5; variation++) {
                int sampleIdx = digit * 5 + variation;
                

                int pixelIdx = 0;
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {

                        double noise = random.nextDouble();
                        if (noise < 0.1 && variation > 0) {
                            inputs[sampleIdx][pixelIdx] = 1.0 - digitPatterns[digit][i][j];
                        } else {
                            inputs[sampleIdx][pixelIdx] = digitPatterns[digit][i][j];
                        }
                        pixelIdx++;
                    }
                }
                

                for (int d = 0; d < numDigits; d++) {
                    targets[sampleIdx][d] = (d == digit) ? 1.0 : 0.0;
                }
            }
        }
        

        NeuralNetwork nn = new NeuralNetwork(0.05);
        nn.addInputLayer(pixelsPerDigit, 64, new ReLUActivation());
        nn.addLayer(32, new ReLUActivation());
        nn.addLayer(numDigits, new SoftmaxActivation());
        
        // Train the network
        System.out.println("Training Simplified MNIST network...");
        nn.fit(inputs, targets, 2000, true);
        

        System.out.println("\nSimplified MNIST Test Results:");
        
        for (int digit = 0; digit < numDigits; digit++) {

            double[] input = new double[pixelsPerDigit];
            int pixelIdx = 0;
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    input[pixelIdx++] = digitPatterns[digit][i][j];
                }
            }
            
            double[] output = nn.predict(input);
            

            int predictedDigit = 0;
            for (int d = 1; d < numDigits; d++) {
                if (output[d] > output[predictedDigit]) {
                    predictedDigit = d;
                }
            }
            
            System.out.printf("Digit %d: Predicted as %d, Confidence: %.2f%%%n",
                digit, predictedDigit, output[predictedDigit] * 100);
            

            System.out.print("Class probabilities: ");
            for (int d = 0; d < numDigits; d++) {
                System.out.printf("[%d]: %.2f%% ", d, output[d] * 100);
            }
            System.out.println();
            

            System.out.println("Pattern:");
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    System.out.print(digitPatterns[digit][i][j] == 1 ? "█" : " ");
                }
                System.out.println();
            }
            System.out.println();
        }
    }
    

    public static void testTimeSeriesPrediction() {
        System.out.println("\n=== Time Series Prediction (Stock Price Simulation) ===");
        

        int dataPoints = 200;
        double[] timeSeriesData = new double[dataPoints];
        

        double baseValue = 100.0;
        

        for (int i = 0; i < dataPoints; i++) {

            double trend = 0.05 * i;
            

            double seasonality = 5.0 * Math.sin(i * 2 * Math.PI / 20.0);
            

            double noise = random.nextGaussian() * 2.0;
            

            timeSeriesData[i] = baseValue + trend + seasonality + noise;
        }
        

        int windowSize = 5;
        int numSamples = dataPoints - windowSize;
        
        double[][] inputs = new double[numSamples][windowSize];
        double[][] targets = new double[numSamples][1];
        

        double minValue = Double.MAX_VALUE;
        double maxValue = Double.MIN_VALUE;
        for (double value : timeSeriesData) {
            minValue = Math.min(minValue, value);
            maxValue = Math.max(maxValue, value);
        }
        double range = maxValue - minValue;
        

        for (int i = 0; i < numSamples; i++) {

            for (int j = 0; j < windowSize; j++) {
                inputs[i][j] = (timeSeriesData[i + j] - minValue) / range;
            }
            

            targets[i][0] = (timeSeriesData[i + windowSize] - minValue) / range;
        }
        

        int trainSize = (int)(numSamples * 0.8);
        int testSize = numSamples - trainSize;
        
        double[][] trainInputs = new double[trainSize][windowSize];
        double[][] trainTargets = new double[trainSize][1];
        double[][] testInputs = new double[testSize][windowSize];
        double[][] testTargets = new double[testSize][1];
        

        for (int i = 0; i < trainSize; i++) {
            System.arraycopy(inputs[i], 0, trainInputs[i], 0, windowSize);
            trainTargets[i][0] = targets[i][0];
        }
        
        for (int i = 0; i < testSize; i++) {
            System.arraycopy(inputs[i + trainSize], 0, testInputs[i], 0, windowSize);
            testTargets[i][0] = targets[i + trainSize][0];
        }
        

        NeuralNetwork nn = new NeuralNetwork(0.01);
        nn.addInputLayer(windowSize, 16, new ReLUActivation());
        nn.addLayer(8, new ReLUActivation());
        nn.addLayer(1, new SigmoidActivation());
        
        System.out.println("Training Time Series Prediction network...");
        nn.fit(trainInputs, trainTargets, 2000, true);
        

        System.out.println("\nTime Series Prediction Test Results:");
        double totalError = 0.0;
        
        System.out.println("Day\tActual\tPredicted\tError");
        System.out.println("---\t------\t---------\t-----");
        
        for (int i = 0; i < testSize; i++) {
            double[] input = testInputs[i];
            double[] output = nn.predict(input);
            

            double actualNormalized = testTargets[i][0];
            double predictedNormalized = output[0];
            
            double actual = actualNormalized * range + minValue;
            double predicted = predictedNormalized * range + minValue;
            
            double error = Math.abs(actual - predicted);
            totalError += error;
            

            if (i % 5 == 0) {
                System.out.printf("%d\t%.2f\t%.2f\t%.2f%n", 
                    i + trainSize + windowSize, actual, predicted, error);
            }
        }
        
        double mae = totalError / testSize;
        System.out.printf("\nMean Absolute Error: %.2f\n", mae);
        System.out.printf("Mean Absolute Percentage Error: %.2f%%\n", (mae / ((maxValue + minValue) / 2)) * 100);
        

        System.out.println("\nFuture Predictions (next 10 days):");
        System.out.println("Day\tPredicted Value");
        System.out.println("---\t--------------");
        

        double[] predictionWindow = new double[windowSize];
        for (int i = 0; i < windowSize; i++) {
            predictionWindow[i] = (timeSeriesData[dataPoints - windowSize + i] - minValue) / range;
        }
        

        for (int i = 0; i < 10; i++) {
            double[] output = nn.predict(predictionWindow);
            double predictedValue = output[0] * range + minValue;
            
            System.out.printf("%d\t%.2f%n", dataPoints + i, predictedValue);
            

            for (int j = 0; j < windowSize - 1; j++) {
                predictionWindow[j] = predictionWindow[j + 1];
            }
            predictionWindow[windowSize - 1] = output[0];
        }
    }
}
