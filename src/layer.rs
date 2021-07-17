pub mod conv2d;
pub mod feed;
pub mod reshape;
pub mod series;
//pub mod parallel;

use std::any::*;

pub trait Layer: Any {
    fn before(&self) -> Vec<usize>;
    fn after(&self) -> Vec<usize>;
    fn forward(&mut self, input: Vec<f32>) -> Vec<f32>;
    fn backward(&mut self, target: Vec<f32>, lr: f32) -> Vec<f32>;
}

pub trait Group: Layer {
    fn push(&mut self, layer: Object);
}

pub enum Object {
    Layer(Box<dyn Layer>),
    Group(Box<dyn Group>),
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

    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        match self {
            Self::Layer(layer) => layer.forward(input),
            Self::Group(group) => group.forward(input),
        }
    }

    fn backward(&mut self, target: Vec<f32>, lr: f32) -> Vec<f32> {
        match self {
            Self::Layer(layer) => layer.backward(target, lr),
            Self::Group(group) => group.backward(target, lr),
        }
    }
}

pub trait Template<L: Layer> {
    fn into(self, before: Vec<usize>) -> L;
}

pub trait Dynamic {
    fn new(before: Vec<usize>) -> Self;
}
