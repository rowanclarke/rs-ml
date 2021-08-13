//pub mod conv2d;
pub mod feed;
//pub mod reshape;
pub mod series;
//pub mod parallel;

<<<<<<< Updated upstream
use super::loss::Loss;
use ndarray::Array2;
use std::any::*;

pub trait Cost {
    fn cost(&self, given: Array2<f32>) -> Array2<f32>;
}

pub trait Layer: Cost + Any {
=======
use super::matrix::Column;
use erased_serde::serialize_trait_object;
use erased_serde::Serialize;
use std::any::Any;

pub trait Layer: Any + Serialize {
>>>>>>> Stashed changes
    fn before(&self) -> Vec<usize>;
    fn after(&self) -> Vec<usize>;
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, next: CostObject, lr: f32);
}

pub trait Group: Layer {
    fn push(&mut self, layer: Object);
}

pub enum Object {
    Layer(Box<dyn Layer>),
    Group(Box<dyn Group>),
}

impl Cost for Object {
    fn cost(&self, given: Array2<f32>) -> Array2<f32> {
        match self {
            Self::Layer(layer) => layer.cost(given),
            Self::Group(group) => group.cost(given),
        }
    }
}

impl Layer for Object {
    fn before(&self) -> Vec<usize> {
        match self {
            Self::Layer(layer) => layer.before(),
            Self::Group(group) => group.before(),
        }
    }

    fn after(&self) -> Vec<usize> {
        match self {
            Self::Layer(layer) => layer.after(),
            Self::Group(group) => group.after(),
        }
    }

    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        match self {
            Self::Layer(layer) => layer.forward(input),
            Self::Group(group) => group.forward(input),
        }
    }

    fn backward(&mut self, dele: CostObject, lr: f32) {
        match self {
            Self::Layer(layer) => layer.backward(dele, lr),
            Self::Group(group) => group.backward(dele, lr),
        }
    }
}

pub enum CostObject {
    Loss(Box<dyn Loss>),
    Layer(Box<dyn Layer>),
    Group(Box<dyn Group>),
}

impl Cost for CostObject {
    fn cost(&self, given: Array2<f32>) -> Array2<f32> {
        match self {
            Self::Loss(loss) => loss.cost(given),
            Self::Layer(layer) => layer.cost(given),
            Self::Group(group) => group.cost(given),
        }
    }
}

pub trait Template<L: Layer> {
    fn into(self, before: Vec<usize>) -> L;
}

<<<<<<< Updated upstream
pub trait Dynamic {
    fn new(before: Vec<usize>) -> Self;
=======
serialize_trait_object!(Layer);

pub trait LayerBuilder {
    fn build(self, before: Vec<usize>) -> Box<dyn Layer>;
>>>>>>> Stashed changes
}
