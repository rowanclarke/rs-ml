use super::super::{
    activation::Activation,
    matrix::{Column, Jacobean, Matrix},
};

pub struct Sigmoid {}

impl Activation for Sigmoid {
    fn activate(vec: Column) -> Column {
        let vec = vec.map(|&x| (-x).exp());
        1.0 / (1.0 + vec)
    }

    fn deactivate(vec: Column) -> Jacobean {
        let vec = Self::activate(vec);
        let vec = &vec * (1.0 - &vec);
        let mut da = Jacobean::zeros((vec.len(), vec.len()));
        for i in 0..vec.len() {
            da[[i, i]] = vec[[i, 0]];
        }
        da
    }
}
