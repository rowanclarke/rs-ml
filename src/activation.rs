pub mod sigmoid;

use super::matrix::{Column, Jacobean};

pub trait Activation: 'static {
    fn activate(vec: Column) -> Column;
    fn deactivate(vec: Column) -> Jacobean;
}
