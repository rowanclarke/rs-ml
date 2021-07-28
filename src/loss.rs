pub mod mse;
//pub mod xent;

use super::layer::Cost;
use ndarray::Array2;
use std::any::Any;

pub trait Loss: Cost + Any {
    fn forward(&self, output: Array2<f32>, target: Array2<f32>) -> f32;
    fn backward(&self, output: Array2<f32>, target: Array2<f32>) -> Array2<f32>;
}

pub trait LossBuilder {
    fn new() -> Self;
}
