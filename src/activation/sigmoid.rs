use super::super::{
    activation::Activation,
    matrix::{Column, Jacobean, Matrix},
};

pub struct Sigmoid {}

impl Activation for Sigmoid {
    fn activate(mut vec: Column) -> Column {
        vec.map(|x| (-x).exp());
        vec.map(|x| 1.0 / (1.0 + x));
        vec
    }

    fn deactivate(vec: Column) -> Jacobean {
        let mut vec = Self::activate(vec);
        vec.map(|x| x * (1.0 - x));
        let mut da = Jacobean::zeros((vec.len(), vec.len()));
        for i in 0..vec.len() {
            da[(i, i)] = vec[i];
        }
        da
    }
}
