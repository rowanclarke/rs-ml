pub mod mse;
pub mod xent;

use super::matrix::Column;
use std::any::Any;

pub trait Loss: Any {
    fn forward(output: Column, target: Column) -> f32;
    fn backward(output: Column, target: Column) -> Column;
}
