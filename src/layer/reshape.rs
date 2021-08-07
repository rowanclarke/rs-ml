use super::super::matrix::Column;
use super::{Layer, LayerBuilder};

pub struct Flatten {}

impl Flatten {
    pub fn new() -> Self {
        Self {}
    }
}

impl LayerBuilder for Flatten {
    fn build(self, before: Vec<usize>) -> Box<dyn Layer> {
        let mut after: usize = 1;
        for i in before.clone() {
            after *= i;
        }
        Box::new(ReshapeLayer {
            before,
            after: vec![after],
        })
    }
}

pub struct Reshape {
    pub shape: Vec<usize>,
}

impl Reshape {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

impl LayerBuilder for Reshape {
    fn build(self, before: Vec<usize>) -> Box<dyn Layer> {
        Box::new(ReshapeLayer {
            before,
            after: self.shape,
        })
    }
}

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

    fn forward(&mut self, input: Column) -> Column {
        input
    }

    fn backward(&mut self, target: Column, lr: f32) -> Column {
        target
    }
}
