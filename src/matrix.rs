use ndarray::{ArrayBase, Dimension, OwnedRepr, StrideShape};
use rand::prelude::*;
use std::fmt;
use std::ops;

#[derive(Clone)]
pub struct Matrix {
    matrix: Vec<f32>,
    shape: (usize, usize),
    transpose: bool,
}

impl Matrix {
    pub fn zeros(shape: (usize, usize)) -> Self {
        Self {
            matrix: vec![0.0; shape.0 * shape.1],
            shape,
            transpose: false,
        }
    }

    pub fn random(shape: (usize, usize)) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            matrix: vec![0.0; shape.0 * shape.1]
                .into_iter()
                .map(|_| rng.gen::<f32>())
                .collect(),
            shape,
            transpose: false,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        if self.transpose {
            (self.shape.1, self.shape.0)
        } else {
            self.shape
        }
    }

    pub fn reshape(&mut self, shape: (usize, usize)) {
        self.shape = shape;
    }

    pub fn transpose(&mut self) {
        self.transpose = !self.transpose;
    }
}

impl<'a, 'b> ops::Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &'b Matrix) -> Matrix {
        let mut result = Matrix::zeros((self.shape().0, rhs.shape().1));
        for i in 0..self.shape().0 {
            for j in 0..rhs.shape().1 {
                for k in 0..self.shape().1 {
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
        let mut result = Column::zeros(self.shape().0);
        for i in 0..self.shape().0 {
            for k in 0..self.shape().1 {
                result[i] += self[(i, k)] * rhs[k];
            }
        }
        result
    }
}

impl<'a> ops::Mul<f32> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f32) -> Matrix {
        let mut result = Matrix::zeros((self.shape.0, self.shape.1));
        for i in 0..self.shape().0 {
            for j in 0..self.shape().1 {
                result[(i, j)] = self[(i, j)] * rhs;
            }
        }
        result
    }
}

impl<'a> ops::SubAssign<&'a Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: &'a Matrix) {
        for i in 0..self.shape().0 {
            for j in 0..self.shape().1 {
                self[(i, j)] -= rhs[(i, j)];
            }
        }
    }
}

impl ops::Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, a: (usize, usize)) -> &f32 {
        &self.matrix[a.0 * self.shape().1 + a.1]
    }
}

impl ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, a: (usize, usize)) -> &mut f32 {
        let s = self.shape().1;
        &mut self.matrix[a.0 * s + a.1]
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.matrix)
    }
}

#[derive(Clone)]
pub struct Column {
    column: Vec<f32>,
}

impl Column {
    pub fn new(column: Vec<f32>) -> Self {
        Self { column }
    }

    pub fn zeros(shape: usize) -> Self {
        Self {
            column: vec![0.0; shape],
        }
    }

    pub fn random(shape: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            column: vec![0.0; shape]
                .into_iter()
                .map(|_| rng.gen::<f32>())
                .collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.column.len()
    }

    pub fn map<F: FnMut(f32) -> f32>(&mut self, mut f: F) {
        for i in 0..self.len() {
            self.column[i] = f(self.column[i]);
        }
    }

    pub fn to_mat(self) -> Matrix {
        Matrix {
            shape: (self.column.len(), 1),
            matrix: self.column,
            transpose: false,
        }
    }

    pub fn to_arr<D: Dimension, Sh: Into<StrideShape<D>>>(
        self,
        shape: Sh,
    ) -> ArrayBase<OwnedRepr<f32>, D> {
        ArrayBase::<OwnedRepr<f32>, D>::from_shape_vec(shape, self.column).unwrap()
    }

    pub fn from_arr<D: Dimension>(array: ArrayBase<OwnedRepr<f32>, D>) -> Column {
        Column::new(array.into_raw_vec())
    }
}

impl<'a, 'b> ops::Add<&'b Column> for &'a Column {
    type Output = Column;

    fn add(self, rhs: &'b Column) -> Column {
        let mut result = Column::zeros(self.column.len());
        for i in 0..self.column.len() {
            result[i] = self.column[i] + rhs.column[i];
        }
        result
    }
}

impl<'a> ops::Mul<f32> for &'a Column {
    type Output = Column;

    fn mul(self, rhs: f32) -> Column {
        let mut result = Column::zeros(self.len());
        for i in 0..self.len() {
            result[i] = self[i] * rhs;
        }
        result
    }
}

impl<'a> ops::SubAssign<&'a Column> for Column {
    fn sub_assign(&mut self, rhs: &'a Column) {
        for i in 0..self.len() {
            self[i] -= rhs[i];
        }
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

impl PartialEq for Column {
    fn eq(&self, other: &Self) -> bool {
        self.column == other.column
    }
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.column)
    }
}

pub type Jacobean = Matrix;
