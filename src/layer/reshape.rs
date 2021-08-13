<<<<<<< Updated upstream
use super::super::loss::Loss;
use super::{Layer, Template};
use std::iter::Product;

pub struct Reshape {
    pub shape: Vec<usize>,
}

impl Reshape {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

impl Template<ReshapeLayer> for Reshape {
    fn into(self, before: Vec<usize>) -> ReshapeLayer {
        ReshapeLayer {
            before,
            after: self.shape,
        }
    }
}
=======
use super::super::matrix::Column;
use super::{Layer, LayerBuilder};
use serde::{Deserialize, Serialize};
>>>>>>> Stashed changes

pub struct Flatten {}

impl Flatten {
    pub fn new() -> Self {
        Self {}
    }
}

impl Template<ReshapeLayer> for Flatten {
    fn into(self, before: Vec<usize>) -> ReshapeLayer {
        let mut after: usize = 1;
        for i in before.clone() {
            after *= i;
        }
        ReshapeLayer {
            before,
            after: vec![after],
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct ReshapeLayer {
    before: Vec<usize>,
    after: Vec<usize>,
}

impl Layer for ReshapeLayer {
    fn before(&self) -> Vec<usize> {
        self.before.clone()
    }

    fn after(&self) -> Vec<usize> {
        self.after.clone()
    }

    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        input
    }

    fn backward(&mut self, target: Vec<f32>, lr: f32) -> Vec<f32> {
        target
    }
}
