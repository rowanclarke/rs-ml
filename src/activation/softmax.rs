use super::super::matrix::{Column, Jacobean};
use super::{Activation, ActivationBuilder};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Softmax {}

#[typetag::serde]
impl Activation for Softmax {
    fn activate(&self, mut vec: Column) -> Column {
        let mut sum = 0.0;
        let mut max = f32::MIN;
        for i in 0..vec.len() {
            max = max.max(vec[i]);
        }
        for i in 0..vec.len() {
            vec[i] = (vec[i] - max).exp();
            sum += vec[i];
        }
        for i in 0..vec.len() {
            vec[i] /= sum;
        }
        vec
    }

    fn deactivate(&self, vec: Column) -> Jacobean {
        let mut del = Jacobean::zeros((vec.len(), vec.len()));
        let vec = self.activate(vec);
        for i in 0..vec.len() {
            for j in 0..vec.len() {
                if i == j {
                    del[(i, j)] = vec[i] * (1.0 - vec[j]);
                } else {
                    del[(i, j)] = -vec[j] * vec[i];
                }
            }
        }
        del
    }
}

impl ActivationBuilder for Softmax {
    fn new() -> Self {
        Softmax {}
    }
}
