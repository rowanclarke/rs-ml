pub mod feed;

use super::loss::Loss;
use super::matrix::{Column, Jacobean};
use std::any::Any;

pub trait Layer: Any {
    fn before(&self) -> Vec<usize>;
    fn after(&self) -> Vec<usize>;
    fn forward(&mut self, input: Column) -> Column;
    fn backward(&mut self, dc_y: Column, lr: f32) -> Column;
}

pub trait LayerBuilder {
    fn build(&self, before: Vec<usize>) -> Box<dyn Layer>;
}
