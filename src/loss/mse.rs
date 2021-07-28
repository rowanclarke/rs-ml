use super::super::matrix::{Column, Jacobean, Matrix};
use super::{Cost, Loss, LossBuilder};
use ndarray::Array2;

pub struct MeanSquaredError {
    output: Column,
    target: Column,
}

impl Cost for MeanSquaredError {
    fn cost(&self, given: Jacobean) -> Jacobean {
        let d = self.backward();
        d.dot(&given)
    }
}

impl Loss for MeanSquaredError {
    fn train(&mut self, output: Column, target: Column) {
        self.output = output;
        self.target = target;
    }

    fn forward(&self) -> f32 {
        let mut sum = 0.0;
        for i in 0..self.output.len() {
            sum += (self.output[[i, 0]] - self.target[[i, 0]]).powf(2.0);
        }
        sum
    }

    fn backward(&self) -> Array2<f32> {
        let mut del = Array2::zeros((1, self.output.len()));
        for i in 0..self.output.len() {
            del[[0, i]] = 2.0 * (self.output[[i, 0]] - self.target[[i, 0]]);
        }
        del
    }
}

impl LossBuilder for MeanSquaredError {
    fn new() -> Self {
        Self {
            output: Column::zeros((0, 0)),
            target: Column::zeros((0, 0)),
        }
    }
}
