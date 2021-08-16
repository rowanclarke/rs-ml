pub mod mse;
pub mod xent;

use super::matrix::Column;
use std::any::Any;

#[typetag::serde(tag = "loss")]
pub trait Loss: Any {
    fn forward(&self, output: Column, target: Column) -> f32;
    fn backward(&self, output: Column, target: Column) -> Column;
}

pub trait LossBuilder {
    fn new() -> Self;
}
