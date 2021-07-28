use super::{Loss, LossBuilder};

pub struct CrossEntropy {}

impl Loss for CrossEntropy {
    fn forward(&self, output: Vec<f32>, target: Vec<f32>) -> f32 {
        let mut sum = 0.0;
        for i in 0..output.len() {
            sum += target[i] * output[i].ln();
        }
        -sum
    }

    fn backward(&self, output: Vec<f32>, target: Vec<f32>) -> Vec<f32> {
        let mut del = vec![0.0; output.len()];
        for i in 0..output.len() {
            del[i] = -target[i] * output[i].recip();
        }
        del
    }
}

impl LossBuilder for CrossEntropy {
    fn new() -> Self {
        Self {}
    }
}
