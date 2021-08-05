use super::layer::{Cost, CostObject, Layer, LayerBuilder};
use super::loss::{Loss, LossBuilder};
use super::matrix::{Column, Matrix};
use std::mem;

pub struct ModelBuilder {
    layers: Vec<Box<dyn LayerBuilder>>,
    before: Vec<Vec<usize>>,
}

impl ModelBuilder {
    pub fn new(before: Vec<usize>) -> Self {
        Self {
            layers: vec![],
            before: vec![before],
        }
    }

    pub fn push_layer(&mut self, template: Box<dyn LayerBuilder>) {
        self.before
            .push(template.after(self.before[self.before.len() - 1].clone()));
        self.layers.push(template);
    }

    pub fn compile<L: LossBuilder + Loss>(mut self, lr: f32) -> Model {
        let mut model = Model::with_loss::<L>(lr, self.before.pop().unwrap());
        for layer in self.layers.into_iter().rev() {
            let cost = mem::replace(&mut model.cost, None).unwrap();
            model.push_layer(layer.build(self.before.pop().unwrap(), cost));
        }
        model
    }
}

pub struct Model {
    cost: Option<CostObject>,
    lr: f32,
    after: Vec<usize>,
}

impl Model {
    pub fn with_loss<L: LossBuilder + Loss>(lr: f32, after: Vec<usize>) -> Self {
        Self {
            cost: Some(CostObject::Loss(Box::new(L::new()))),
            lr,
            after,
        }
    }

    pub fn push_layer(&mut self, layer: Box<dyn Layer>) {
        self.after = layer.before();
        self.cost = Some(CostObject::Layer(layer));
    }

    pub fn train(&mut self, inputs: &[Column], targets: &[Column], epochs: u32) {
        for _ in 0..epochs {
            for x in 0..inputs.len() {
                match self.cost.as_mut().unwrap() {
                    CostObject::Layer(layer) => {
                        layer.train(inputs[x].clone(), targets[x].clone(), self.lr)
                    }
                    CostObject::Loss(_) => (),
                };
            }
        }
    }

    pub fn test(&mut self, inputs: &[Column]) {
        for x in 0..inputs.len() {
            match self.cost.as_mut().unwrap() {
                CostObject::Layer(layer) => layer.test(inputs[x].clone()),
                CostObject::Loss(_) => (),
            };
        }
    }
}
