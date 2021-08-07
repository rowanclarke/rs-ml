use super::super::matrix::{Column, Jacobean};
use super::Activation;

pub struct ReLU {}

impl Activation for ReLU {
    fn activate(mut vec: Column) -> Column {
        for i in 0..vec.len() {
            if vec[i] < 0.0 {
                vec[i] = 0.0;
            }
        }
        vec
    }

    fn deactivate(vec: Column) -> Jacobean {
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
