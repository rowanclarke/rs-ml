use super::layer::{Layer, LayerBuilder};
use super::loss::Loss;
use super::matrix::{Column, Matrix};
use std::marker::PhantomData;
use std::mem;

pub struct ModelBuilder {
    layers: Vec<Box<dyn Layer>>,
    before: Vec<usize>,
}

impl ModelBuilder {
    pub fn new(before: Vec<usize>) -> Self {
        Self {
            layers: vec![],
            before,
        }
    }

    pub fn push_layer<L: LayerBuilder>(&mut self, template: L) {
        self.layers.push(template.build(self.before.clone()));
    }

    pub fn compile<L: Loss>(mut self, lr: f32) -> Model<L> {
        Model::<L> {
            lr,
            layers: self.layers,
            phantom: PhantomData,
        }
    }
}

pub struct Model<L: Loss> {
    lr: f32,
    layers: Vec<Box<dyn Layer>>,
    phantom: PhantomData<L>,
}

impl<L: Loss> Model<L> {
    pub fn train(&mut self, inputs: &[Column], targets: &[Column], epochs: u32) {
        for e in 0..epochs {
            for i in 0..inputs.len() {
                let mut x = inputs[i].clone();
                for l in 0..self.layers.len() {
                    x = self.layers[l].forward(x);
                }
                let mut dc_y = L::backward(x, targets[i].clone());
                for l in (0..self.layers.len()).rev() {
                    dc_y = self.layers[l].backward(dc_y, self.lr);
                }
            }
        }
    }

    pub fn test(&mut self, inputs: &[Column]) {
        for i in 0..inputs.len() {
            let mut x = inputs[i].clone();
            for l in 0..self.layers.len() {
                x = self.layers[l].forward(x);
            }
            println!("{}", x);
        }
    }
}
