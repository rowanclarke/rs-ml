use super::Activation;

pub struct ReLU {}

impl Activation for ReLU {
    fn activate(mut vec: Vec<f32>) -> Vec<f32> {
        for i in 0..vec.len() {
            if vec[i] < 0.0 {
                vec[i] = 0.0;
            }
        }
        vec
    }

    fn deactivate(mut vec: Vec<f32>) -> Vec<f32> {
        for i in 0..vec.len() {
            if vec[i] < 0.0 {
                vec[i] = 0.0;
            } else {
                vec[i] = 1.0;
            }
        }
        vec
    }
}
