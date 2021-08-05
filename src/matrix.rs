use ndarray::{Array2, ArrayD, IxDyn};
use std::fmt;
use std::ops;

#[derive(Clone)]
pub struct Matrix {
    matrix: Vec<f32>,
    shape: (usize, usize),
}

impl Matrix {
    pub fn zeros(shape: (usize, usize)) -> Self {
        Self {
            matrix: vec![0.0; shape.0 * shape.1],
            shape,
        }
    }

    pub fn random(shape: (usize, usize)) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            matrix: vec![0.0; shape.0 * shape.1]
                .into_iter()
                .map(|x| 1.0)
                .collect(),
            shape,
        }
    }
}

impl<'a, 'b> ops::Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &'b Matrix) -> Matrix {
        let mut result = Matrix::zeros((self.shape.0, rhs.shape.1));
        for i in 0..self.shape.0 {
            for j in 0..rhs.shape.1 {
                for k in 0..self.shape.1 {
                    result[(i, j)] += self[(i, k)] * rhs[(k, j)];
                }
            }
        }
        result
    }
}

impl<'a, 'b> ops::Mul<&'b Column> for &'a Matrix {
    type Output = Column;

    fn mul(self, rhs: &'b Column) -> Column {
        let mut result = Column::zeros(self.shape.0);
        for i in 0..self.shape.0 {
            for k in 0..self.shape.1 {
                result[i] += self[(i, k)] * rhs[k];
            }
        }
        result
    }
}

impl ops::Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, a: (usize, usize)) -> &f32 {
        &self.matrix[a.0 * self.shape.1 + a.1]
    }
}

impl ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, a: (usize, usize)) -> &mut f32 {
        &mut self.matrix[a.0 * self.shape.1 + a.1]
    }
}

#[derive(Clone)]
pub struct Column {
    column: Vec<f32>,
}

impl Column {
    pub fn zeros(shape: usize) -> Self {
        Self {
            column: vec![0.0; shape],
        }
    }

    pub fn random(shape: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            column: vec![0.0; shape].into_iter().map(|x| 1.0).collect(),
        }
    }

    pub fn to_mat(self) -> Matrix {
        Matrix {
            shape: (self.column.len(), 1),
            matrix: self.column,
        }
    }

    pub fn to_shape(self, ix: IxDyn) -> ArrayD<f32> {
        ArrayD::<f32>::from_shape_vec(ix, self.column).unwrap()
    }
}

impl<'a, 'b> ops::Add<&'b Column> for &'a Column {
    type Output = Column;

    fn add(self, rhs: &'b Column) -> Column {
        let result = Column::zeros(self.column.len());
        for i in 0..self.column.len() {
            result[i] = self.column[i] + rhs.column[i];
        }
        result
    }
}

impl ops::Index<usize> for Column {
    type Output = f32;

    fn index(&self, a: usize) -> &f32 {
        &self.column[a]
    }
}

impl ops::IndexMut<usize> for Column {
    fn index_mut(&mut self, a: usize) -> &mut f32 {
        &mut self.column[a]
    }
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.column)
    }
}

pub type Jacobean = Matrix;
