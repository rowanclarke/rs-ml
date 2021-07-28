//pub mod relu;
pub mod sigmoid;
//pub mod softmax;

pub use ndarray::Array2;

pub trait Activation: 'static {
    fn activate(vec: Array2<f32>) -> Array2<f32>;
    fn deactivate(vec: Array2<f32>) -> Array2<f32>;
}
