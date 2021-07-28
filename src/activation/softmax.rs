use super::Activation;

pub struct Softmax {}

impl Activation for Softmax {
    fn activate(mut vec: Vec<f32>) -> Vec<f32> {
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

    fn deactivate(vec: Vec<f32>) -> Vec<f32> {
        let mut del = vec![0.0; vec.len() * vec.len()];
        let vec = Self::activate(vec);
        for i in 0..vec.len() {
            for j in 0..vec.len() {
                if i == j {
                    del[i * vec.len() + j] = vec[i] * (1.0 - vec[j]);
                } else {
                    del[i * vec.len() + j] = -vec[j] * vec[i];
                }
            }
        }
        del
    }
}
