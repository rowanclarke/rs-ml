pub mod feed;
pub mod parallel;
pub mod series;

use std::any::*;

pub trait Layer: Any {
    fn forward(&mut self, input: Vec<f32>) -> Vec<f32>;
    fn backward(&mut self);
}
