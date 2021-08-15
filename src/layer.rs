pub mod conv2d;
pub mod feed;
pub mod reshape;

use super::matrix::Column;
use erased_serde::serialize_trait_object;
use std::any::Any;

#[typetag::serde(tag = "type")]
pub trait Layer: Any {
    fn before(&self) -> Vec<usize>;
    fn after(&self) -> Vec<usize>;
    fn forward(&mut self, input: Column) -> Column;
    fn backward(&mut self, dc_y: Column, lr: f32) -> Column;
}

pub trait LayerBuilder {
    fn build(self, before: Vec<usize>) -> Box<dyn Layer>;
}
