use super::{Cost, Loss, LossBuilder};
use ndarray::Array2;

pub struct MeanSquaredError {}

impl Cost for MeanSquaredError {
    fn cost(&self, given: Array2<f32>) -> Array2<f32> {
        given
    }
}

impl Loss for MeanSquaredError {
    fn forward(&self, output: Array2<f32>, target: Array2<f32>) -> f32 {
        let mut sum = 0.0;
        for i in 0..output.len() {
            sum += (output[[i, 0]] - target[[i, 0]]).powf(2.0);
        }
        sum / output.len() as f32
    }

    fn backward(&self, output: Array2<f32>, target: Array2<f32>) -> Array2<f32> {
        let mut del = Array2::zeros((1, output.len()));
        for i in 0..output.len() {
            del[[i, 0]] = 2.0 / output.len() as f32 * (output[[i, 0]] - target[[i, 0]]);
        }
        del
    }
}

impl LossBuilder for MeanSquaredError {
    fn new() -> Self {
        Self {}
    }
}
