use super::Activation;
use ndarray::Array2;

pub struct Sigmoid {}

impl Activation for Sigmoid {
    fn activate(vec: Array2<f32>) -> Array2<f32> {
        1.0 / (1.0 + vec.map(|x| x.exp()))
    }

    fn deactivate(vec: Array2<f32>) -> Array2<f32> {
        let vec = Self::activate(vec);
        let vec = vec * (1.0 - vec);
        let mut del = Array2::zeros((vec.len(), vec.len()));
        for i in 0..vec.len() {
            del[[i, i]] = vec[[i, 0]];
        }
        del
    }
}
