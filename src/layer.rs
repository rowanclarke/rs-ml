pub mod conv2d;
pub mod feed;

use super::loss::Loss;
use super::matrix::{Column, Jacobean};
use std::any::Any;

pub trait Cost {
    fn cost(&self, given: Jacobean) -> Jacobean;
}

pub trait Layer: Cost + Any {
    fn before(&self) -> Vec<usize>;
    fn after(&self) -> Vec<usize>;
    fn train(&mut self, input: Column, target: Column, lr: f32);
    fn test(&mut self, input: Column);
    fn forward(&mut self, input: Column);
    fn backward(&mut self, lr: f32);
}

pub trait LayerBuilder {
    fn build(&self, after: Vec<usize>, other: CostObject) -> Box<dyn Layer>;
    fn after(&self, before: Vec<usize>) -> Vec<usize>;
}

pub enum CostObject {
    Loss(Box<dyn Loss>),
    Layer(Box<dyn Layer>),
}

impl Cost for CostObject {
    fn cost(&self, given: Jacobean) -> Jacobean {
        match self {
            Self::Loss(loss) => loss.cost(given),
            Self::Layer(layer) => layer.cost(given),
        }
    }
}
