pub mod mse;

use super::matrix::{Column, Jacobean};
use std::any::Any;

pub trait Loss: Any {
    fn forward(output: Column, target: Column) -> f32;
    fn backward(output: Column, target: Column) -> Column;
}
