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
        let mut del = vec![0.0; vec.len() * vec.len()];
        for i in 0..vec.len() {
            for j in 0..vec.len() {
                if i == j {
                    if vec[i] < 0.0 {
                        del[i * vec.len() + j] = 0.0;
                    } else {
                        del[i * vec.len() + j] = 1.0;
                    }
                }
            }
        }
        vec
    }
}
