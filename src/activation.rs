pub mod relu;
pub mod sigmoid;
pub mod softmax;

use super::matrix::{Column, Jacobean};
use std::any::Any;

#[typetag::serde(tag = "activation")]
pub trait Activation: Any {
    fn activate(&self, vec: Column) -> Column;
    fn deactivate(&self, vec: Column) -> Jacobean;
}

pub trait ActivationBuilder {
    fn new() -> Self;
}
