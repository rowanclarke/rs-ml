use super::Activation;

pub struct Sigmoid {}

impl Activation for Sigmoid {
    fn activate(vec: Vec<f32>) -> Vec<f32> {
        vec.into_iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
    }

    fn deactivate(vec: Vec<f32>) -> Vec<f32> {
        Self::activate(vec)
            .into_iter()
            .map(|x| x * (1.0 - x))
            .collect()
    }
}
