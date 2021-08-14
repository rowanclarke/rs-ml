pub mod conv2d;
pub mod feed;
pub mod reshape;

use super::matrix::Column;
use erased_serde::serialize_trait_object;
use erased_serde::Serialize;
use std::any::Any;

pub trait Layer: Any + Serialize {
    fn before(&self) -> Vec<usize>;
    fn after(&self) -> Vec<usize>;
    fn forward(&mut self, input: Column) -> Column;
    fn backward(&mut self, dc_y: Column, lr: f32) -> Column;
}

serialize_trait_object!(Layer);

pub trait LayerBuilder {
    fn build(self, before: Vec<usize>) -> Box<dyn Layer>;
}
