pub mod relu;
pub mod sigmoid;
pub mod softmax;

pub trait Activation: 'static {
    fn activate(vec: Vec<f32>) -> Vec<f32>;
    fn deactivate(vec: Vec<f32>) -> Vec<f32>;
}
