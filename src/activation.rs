pub mod relu;
pub mod sigmoid;
pub mod softmax;

use super::matrix::{Column, Jacobean};

pub trait Activation: 'static {
    fn activate(vec: Column) -> Column;
    fn deactivate(vec: Column) -> Jacobean;
}
