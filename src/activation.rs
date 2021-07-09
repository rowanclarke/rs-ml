pub mod sigmoid;

pub trait Activation {
    fn activate(vec: Vec<f32>) -> Vec<f32>;
    fn deactivate(vec: Vec<f32>) -> Vec<f32>;
}
