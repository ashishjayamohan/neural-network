import java.util.function.DoubleUnaryOperator;
import java.util.Random;

public class Matrix {
  private static final Random rand = new Random();

  public final int rows;
  public final int cols;
  private final double[][] data;

  public Matrix(int rows, int cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = new double[rows][cols];
  }

  public static Matrix fromArray(double[] array) {
    Matrix result = new Matrix(array.length, 1);
    for (int i = 0; i < array.length; i++) {
      result.data[i][0] = array[i];
    }
    return result;
  }

  public double[] toArray() {
    double[] result = new double[rows * cols];
    int idx = 0;
    for (double[] row : data) {
      for (double val : row) {
        result[idx++] = val;
      }
    }
    return result;
  }

  public void randomize() {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = rand.nextDouble() * 2 - 1;
      }
    }
  }

  public Matrix copy() {
    Matrix result = new Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      System.arraycopy(this.data[i], 0, result.data[i], 0, cols);
    }
    return result;
  }

  public static Matrix dot(Matrix a, Matrix b) {
    if (a.cols != b.rows) {
      throw new IllegalArgumentException("Incompatible matrix sizes for dot product");
    }
    Matrix result = new Matrix(a.rows, b.cols);
    for (int i = 0; i < result.rows; i++) {
      for (int j = 0; j < result.cols; j++) {
        double sum = 0.0;
        for (int k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }

  public Matrix add(Matrix other) {
    checkSizeMatch(other);
    Matrix result = new Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[i][j] = this.data[i][j] + other.data[i][j];
      }
    }
    return result;
  }

  public Matrix subtract(Matrix other) {
    checkSizeMatch(other);
    Matrix result = new Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[i][j] = this.data[i][j] - other.data[i][j];
      }
    }
    return result;
  }

  public void subtractInPlace(Matrix other) {
    checkSizeMatch(other);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this.data[i][j] -= other.data[i][j];
      }
    }
  }

  public Matrix multiply(double scalar) {
    Matrix result = new Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[i][j] = this.data[i][j] * scalar;
      }
    }
    return result;
  }

  public static Matrix transpose(Matrix m) {
    Matrix result = new Matrix(m.cols, m.rows);
    for (int i = 0; i < m.rows; i++) {
      for (int j = 0; j < m.cols; j++) {
        result.data[j][i] = m.data[i][j];
      }
    }
    return result;
  }

  public Matrix map(DoubleUnaryOperator func) {
    Matrix result = new Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[i][j] = func.applyAsDouble(this.data[i][j]);
      }
    }
    return result;
  }

  public static Matrix hadamard(Matrix a, Matrix b) {
    a.checkSizeMatch(b);
    Matrix result = new Matrix(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
      for (int j = 0; j < a.cols; j++) {
        result.data[i][j] = a.data[i][j] * b.data[i][j];
      }
    }
    return result;
  }

  private void checkSizeMatch(Matrix other) {
    if (this.rows != other.rows || this.cols != other.cols) {
      throw new IllegalArgumentException("Matrix size mismatch");
    }
  }

  public void print() {
    for (double[] row : data) {
      for (double val : row) {
        System.out.printf("%.4f ", val);
      }
      System.out.println();
    }
  }
  
  public double get(int row, int col) {
    return data[row][col];
  }
  
  public void set(int row, int col, double value) {
    data[row][col] = value;
  }
}