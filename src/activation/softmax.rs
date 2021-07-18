use super::Activation;

pub struct Softmax {}

impl Activation for Softmax {
    fn activate(mut vec: Vec<f32>) -> Vec<f32> {
        let mut sum = 0.0;
        for i in 0..vec.len() {
            vec[i] = vec[i].exp();
            sum += vec[i];
        }
        for i in 0..vec.len() {
            vec[i] /= sum;
        }
        vec
    }

    fn deactivate(vec: Vec<f32>) -> Vec<f32> {
        let mut vec = Self::activate(vec);
        for i in 0..vec.len() {
            vec[i] = vec[i] * (1.0 - vec[i]);
        }
        vec
    }
}
