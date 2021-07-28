pub mod mse;

use super::layer::Cost;
use super::matrix::{Column, Jacobean};
use std::any::Any;

pub trait Loss: Cost + Any {
    fn train(&mut self, output: Column, target: Column);
    fn forward(&self) -> f32;
    fn backward(&self) -> Jacobean;
}

pub trait LossBuilder {
    fn new() -> Self;
}
