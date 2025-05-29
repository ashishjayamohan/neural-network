use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::fmt;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![0.0; rows * cols];
        Matrix { rows, cols, data }
    }

    pub fn from_array(array: &[f64]) -> Self {
        let rows = array.len();
        let cols = 1;
        let mut data = vec![0.0; rows * cols];
        data.copy_from_slice(array);
        Matrix { rows, cols, data }
    }

    pub fn to_array(&self) -> Vec<f64> {
        self.data.clone()
    }

    pub fn randomize(&mut self) {
        // Use a thread-safe RNG for parallel code
        let seed = rand::random::<u64>();
        
        // Create a new vector with random values
        let random_data: Vec<f64> = (0..self.data.len())
            .into_par_iter()
            .map(|_| {
                let mut local_rng = SmallRng::seed_from_u64(seed.wrapping_add(rand::random::<u64>()));
                local_rng.gen::<f64>() * 2.0 - 1.0
            })
            .collect();
        
        // Replace the data with random values
        self.data = random_data;
    }

    pub fn dot(a: &Matrix, b: &Matrix) -> Result<Matrix, &'static str> {
        if a.cols != b.rows {
            return Err("Incompatible matrix sizes for dot product");
        }
        
        let mut result = Matrix::new(a.rows, b.cols);
        
        // Lower threshold for parallelization in tests
        // This makes small matrix operations faster in tests
        let threshold = if cfg!(test) { 100 } else { 1000 };
        
        // Parallel execution for larger matrices, sequential for smaller ones
        if a.rows * b.cols > threshold {
            // Create a copy of the data for thread safety
            let a_data = a.data.clone();
            let b_data = b.data.clone();
            let a_cols = a.cols;
            let b_cols = b.cols;
            
            // Use direct indexing for better performance
            let result_data = (0..a.rows)
                .into_par_iter()
                .flat_map(|i| {
                    let mut row_result = vec![0.0; b_cols];
                    let i_offset = i * a_cols;
                    
                    for j in 0..b_cols {
                        let mut sum = 0.0;
                        for k in 0..a_cols {
                            sum += a_data[i_offset + k] * b_data[k * b_cols + j];
                        }
                        row_result[j] = sum;
                    }
                    
                    row_result
                })
                .collect();
            
            // Replace the entire data vector at once
            result.data = result_data;
        } else {
            // For small matrices, use direct indexing for better performance
            let a_data = &a.data;
            let b_data = &b.data;
            let a_cols = a.cols;
            let b_cols = b.cols;
            
            for i in 0..a.rows {
                for j in 0..b.cols {
                    let mut sum = 0.0;
                    let i_offset = i * a_cols;
                    
                    for k in 0..a_cols {
                        sum += a_data[i_offset + k] * b_data[k * b_cols + j];
                    }
                    
                    result.data[i * b_cols + j] = sum;
                }
            }
        }
        
        Ok(result)
    }

    pub fn add(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        self.check_size_match(other)?;
        let mut result = Matrix::new(self.rows, self.cols);
        
        // Create a parallel iterator to process the data
        let self_data = &self.data;
        let other_data = &other.data;
        
        let processed: Vec<f64> = (0..self.data.len())
            .into_par_iter()
            .map(|i| self_data[i] + other_data[i])
            .collect();
        
        result.data = processed;
        Ok(result)
    }

    pub fn subtract(&self, other: &Matrix) -> Result<Matrix, &'static str> {
        self.check_size_match(other)?;
        let mut result = Matrix::new(self.rows, self.cols);
        
        // Create a parallel iterator to process the data
        let self_data = &self.data;
        let other_data = &other.data;
        
        let processed: Vec<f64> = (0..self.data.len())
            .into_par_iter()
            .map(|i| self_data[i] - other_data[i])
            .collect();
        
        result.data = processed;
        Ok(result)
    }

    pub fn multiply(&self, scalar: f64) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        
        // Create a parallel iterator to process the data
        let self_data = &self.data;
        
        let processed: Vec<f64> = self_data
            .par_iter()
            .map(|&val| val * scalar)
            .collect();
        
        result.data = processed;
        result
    }

    pub fn transpose(m: &Matrix) -> Matrix {
        let mut result = Matrix::new(m.cols, m.rows);
        
        // Lower threshold for parallelization in tests
        let threshold = if cfg!(test) { 100 } else { 1000 };
        
        // Use parallel execution for larger matrices
        if m.rows * m.cols > threshold {
            // Direct indexing for better performance
            let m_data = &m.data;
            let m_cols = m.cols;
            let result_cols = result.cols;
            
            // Create transposed data in parallel and then assign it
            let transposed_data: Vec<(usize, f64)> = (0..m.rows)
                .into_par_iter()
                .flat_map(|i| {
                    let mut row_data = Vec::with_capacity(m.cols);
                    for j in 0..m.cols {
                        let src_idx = i * m_cols + j;
                        let dst_idx = j * result_cols + i;
                        row_data.push((dst_idx, m_data[src_idx]));
                    }
                    row_data
                })
                .collect();
            
            // Apply the transposed values
            for (idx, val) in transposed_data {
                result.data[idx] = val;
            }
        } else {
            // For small matrices, use direct indexing
            let m_data = &m.data;
            let m_cols = m.cols;
            let result_cols = result.cols;
            
            for i in 0..m.rows {
                for j in 0..m.cols {
                    result.data[j * result_cols + i] = m_data[i * m_cols + j];
                }
            }
        }
        
        result
    }

    pub fn map<F>(&self, func: F) -> Matrix
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        let mut result = Matrix::new(self.rows, self.cols);
        
        // Lower threshold for parallelization in tests
        let threshold = if cfg!(test) { 100 } else { 1000 };
        
        // Use parallel execution only for larger matrices
        if self.data.len() > threshold {
            // Create a parallel iterator to process the data
            let processed: Vec<f64> = self.data
                .par_iter()
                .map(|&val| func(val))
                .collect();
            
            result.data = processed;
        } else {
            // For small matrices, use direct indexing without parallelism
            for (i, &val) in self.data.iter().enumerate() {
                result.data[i] = func(val);
            }
        }
        
        result
    }
    
    pub fn hadamard(a: &Matrix, b: &Matrix) -> Result<Matrix, &'static str> {
        a.check_size_match(b)?;
        let mut result = Matrix::new(a.rows, a.cols);
        
        // Lower threshold for parallelization in tests
        let threshold = if cfg!(test) { 100 } else { 1000 };
        
        // Use parallel execution only for larger matrices
        if a.data.len() > threshold {
            // Create a parallel iterator to process the data
            let a_data = &a.data;
            let b_data = &b.data;
            
            // Use SIMD-friendly processing in chunks
            let processed: Vec<f64> = (0..a.data.len())
                .into_par_iter()
                .map(|i| a_data[i] * b_data[i])
                .collect();
            
            result.data = processed;
        } else {
            // For small matrices, use direct indexing without parallelism
            for i in 0..a.data.len() {
                result.data[i] = a.data[i] * b.data[i];
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

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }
    
    pub fn apply_in_place<F>(&mut self, func: F)
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        // Lower threshold for parallelization in tests
        let threshold = if cfg!(test) { 100 } else { 1000 };
        
        // Use parallel execution only for larger matrices
        if self.data.len() > threshold {
            // Create a new vector with processed values
            let processed: Vec<f64> = self.data
                .par_iter()
                .map(|&val| func(val))
                .collect();
            
            // Replace the data with processed values
            self.data = processed;
        } else {
            // For small matrices, modify in place without parallelism
            for i in 0..self.data.len() {
                self.data[i] = func(self.data[i]);
            }
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:.4} ", self.get(i, j))?;
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
    
    #[test]
    fn test_parallel_operations() {
        let size = 100;
        let mut a = Matrix::new(size, size);
        let mut b = Matrix::new(size, size);
        
        // Fill with test data
        for i in 0..size {
            for j in 0..size {
                a.set(i, j, 1.0);
                b.set(i, j, 2.0);
            }
        }
        
        // Test parallel add
        let sum = a.add(&b).unwrap();
        assert_eq!(sum.get(50, 50), 3.0);
        
        // Test parallel multiply
        let scaled = a.multiply(5.0);
        assert_eq!(scaled.get(50, 50), 5.0);
        
        // Test parallel map
        let mapped = a.map(|x| x * 10.0);
        assert_eq!(mapped.get(50, 50), 10.0);
    }
}