use rand::Rng;
use std::fmt;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![vec![0.0; cols]; rows];
        Matrix { rows, cols, data }
    }

    pub fn from_array(array: &[f64]) -> Self {
        let rows = array.len();
        let mut data = vec![vec![0.0; 1]; rows];
        for i in 0..rows {
            data[i][0] = array[i];
        }
        Matrix { rows, cols: 1, data }
    }

    pub fn to_array(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.rows * self.cols);
        for row in &self.data {
            for &val in row {
                result.push(val);
            }
        }
        result
    }

    pub fn randomize(&mut self) {
        let mut rng = rand::thread_rng();
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0;
            }
        }
    }

    pub fn dot(a: &Matrix, b: &Matrix) -> Result<Matrix, &'static str> {
        if a.cols != b.rows {
            return Err("Incompatible matrix sizes for dot product");
        }
        let mut result = Matrix::new(a.rows, b.cols);
        for i in 0..result.rows {
            for j in 0..result.cols {
                let mut sum = 0.0;
                for k in 0..a.cols {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        Ok(result)
    }

    pub fn add(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        self.check_size_match(other)?;
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        Ok(result)
    }

    pub fn subtract(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        self.check_size_match(other)?;
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        Ok(result)
    }

    pub fn multiply(&self, scalar: f64) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] * scalar;
            }
        }
        result
    }

    pub fn transpose(m: &Matrix) -> Matrix {
        let mut result = Matrix::new(m.cols, m.rows);
        for i in 0..m.rows {
            for j in 0..m.cols {
                result.data[j][i] = m.data[i][j];
            }
        }
        result
    }

    pub fn map<F>(&self, func: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = func(self.data[i][j]);
            }
        }
        result
    }

    pub fn hadamard(a: &Matrix, b: &Matrix) -> Result<Matrix, &'static str> {
        a.check_size_match(b)?;
        let mut result = Matrix::new(a.rows, a.cols);
        for i in 0..a.rows {
            for j in 0..a.cols {
                result.data[i][j] = a.data[i][j] * b.data[i][j];
            }
        }
        Ok(result)
    }

    fn check_size_match(&self, other: &Matrix) -> Result<(), &'static str> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrix size mismatch");
        }
        Ok(())
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row][col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row][col] = value;
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row in &self.data {
            for &val in row {
                write!(f, "{:.4} ", val)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m = Matrix::new(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.get(0, 0), 0.0);
    }

    #[test]
    fn test_from_array() {
        let arr = [1.0, 2.0, 3.0];
        let m = Matrix::from_array(&arr);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 1);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 0), 2.0);
        assert_eq!(m.get(2, 0), 3.0);
    }

    #[test]
    fn test_to_array() {
        let mut m = Matrix::new(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);
        let arr = m.to_array();
        assert_eq!(arr, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dot_product() {
        let mut a = Matrix::new(2, 3);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(0, 2, 3.0);
        a.set(1, 0, 4.0);
        a.set(1, 1, 5.0);
        a.set(1, 2, 6.0);

        let mut b = Matrix::new(3, 2);
        b.set(0, 0, 7.0);
        b.set(0, 1, 8.0);
        b.set(1, 0, 9.0);
        b.set(1, 1, 10.0);
        b.set(2, 0, 11.0);
        b.set(2, 1, 12.0);

        let result = Matrix::dot(&a, &b).unwrap();
        assert_eq!(result.get(0, 0), 58.0);
        assert_eq!(result.get(0, 1), 64.0);
        assert_eq!(result.get(1, 0), 139.0);
        assert_eq!(result.get(1, 1), 154.0);
    }
}
