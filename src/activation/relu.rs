use super::super::matrix::{Column, Jacobean};
use super::{Activation, ActivationBuilder};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct ReLU {}

#[typetag::serde]
impl Activation for ReLU {
    fn activate(&self, mut vec: Column) -> Column {
        for i in 0..vec.len() {
            if vec[i] < 0.0 {
                vec[i] = 0.0;
            }
        }
        vec
    }

    fn deactivate(&self, vec: Column) -> Jacobean {
        let mut del = Jacobean::zeros((vec.len(), vec.len()));
        for i in 0..vec.len() {
            for j in 0..vec.len() {
                if i == j {
                    if vec[i] < 0.0 {
                        del[(i, j)] = 0.0;
                    } else {
                        del[(i, j)] = 1.0;
                    }
                }
            }
        }
        del
    }
}

impl ActivationBuilder for ReLU {
    fn new() -> Self {
        ReLU {}
    }
}
