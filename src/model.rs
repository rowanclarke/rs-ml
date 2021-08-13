use super::layer::{Layer, LayerBuilder};
use super::loss::Loss;
use super::matrix::Column;
use std::marker::PhantomData;

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
        let layer = template.build(self.before.clone());
        self.before = layer.after();
        self.layers.push(layer);
    }

    pub fn compile<L: Loss>(self, lr: f32) -> Model<L> {
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
    pub fn train(&mut self, inputs: Box<[Column]>, targets: Box<[Column]>, epochs: u32) {
        for e in 0..epochs {
            for i in 0..inputs.len() {
                let mut x = inputs[i].clone();
                for l in 0..self.layers.len() {
                    x = self.layers[l].forward(x);
                }
                let cost = L::forward(x.clone(), targets[i].clone());
                println!("{}: cost: {}", e, cost);
                let mut dc_y = L::backward(x, targets[i].clone());
                for l in (0..self.layers.len()).rev() {
                    dc_y = self.layers[l].backward(dc_y, self.lr);
                }
            }
        }
    }

    pub fn test(&mut self, inputs: Box<[Column]>) {
        for i in 0..inputs.len() {
            let mut x = inputs[i].clone();
            for l in 0..self.layers.len() {
                x = self.layers[l].forward(x);
            }
            println!("{}", x);
        }
    }
}
