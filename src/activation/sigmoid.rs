use super::super::matrix::{Column, Jacobean};
use super::{Activation, ActivationBuilder};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Sigmoid {}

#[typetag::serde]
impl Activation for Sigmoid {
    fn activate(&self, mut vec: Column) -> Column {
        vec.map(|x| (-x).exp());
        vec.map(|x| 1.0 / (1.0 + x));
        vec
    }

    fn deactivate(&self, vec: Column) -> Jacobean {
        let mut vec = self.activate(vec);
        vec.map(|x| x * (1.0 - x));
        let mut da = Jacobean::zeros((vec.len(), vec.len()));
        for i in 0..vec.len() {
            da[(i, i)] = vec[i];
        }
        da
    }
}

impl ActivationBuilder for Sigmoid {
    fn new() -> Self {
        Sigmoid {}
    }
}
