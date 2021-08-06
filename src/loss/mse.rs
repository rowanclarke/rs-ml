use super::super::matrix::{Column, Jacobean, Matrix};
use super::Loss;
use ndarray::Array2;

pub struct MeanSquaredError {
    output: Column,
    target: Column,
}

impl Loss for MeanSquaredError {
    fn forward(output: Column, target: Column) -> f32 {
        let mut sum = 0.0;
        for i in 0..output.len() {
            sum += (output[i] - target[i]).powf(2.0);
        }
        sum
    }

    fn backward(output: Column, target: Column) -> Column {
        let mut result = Column::zeros(output.len());
        for i in 0..output.len() {
            result[i] = 2.0 * (output[i] - target[i]);
        }
        result
    }
}
