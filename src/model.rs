use super::layer::{series::Series, Cost, CostObject, Dynamic, Group, Layer, Object, Template};
use super::loss::{mse::MeanSquaredError, Loss, LossBuilder};
use ndarray::prelude::Array2;
use std::mem;

pub struct Model {
    stack: Vec<Box<dyn Group>>,
    series: Box<Series>,
    loss: Box<dyn Loss>,
    lr: f32,
}

impl Model {
    pub fn new(size: Vec<usize>) -> Self {
        Self {
            stack: Vec::new(),
            series: Box::new(Series::new(size)),
            loss: Box::new(MeanSquaredError::new()),
            lr: 0.01,
        }
    }

    fn last(&mut self) -> &mut dyn Group {
        let len = self.stack.len();
        if len == 0 {
            return self.series.as_mut();
        } else {
            return self.stack[len].as_mut();
        }
    }

    pub fn push_group<G: Group + Dynamic>(&mut self) {
        // Pushed group must go onto stack
        let size = self.last().after();
        self.stack.push(Box::new(G::new(size)));
    }

    pub fn pop_group(&mut self) {
        // Popped group must be from stack
        let group = self.stack.pop().unwrap();
        self.last().push(Object::Group(group));
    }

    pub fn push_layer<L: Layer, T: Template<L>>(&mut self, template: T) {
        let last = self.last();
        let before = last.before();
        last.push(Object::Layer(Box::new(template.into(before))));
    }

    pub fn compile<L: LossBuilder + Loss>(&mut self, lr: f32) {
        self.loss = Box::new(L::new());
        self.lr = lr;
    }

    pub fn train(&mut self, inputs: &[Array2<f32>], targets: &[Array2<f32>], epochs: u32) {
        for e in 0..epochs {
            for x in 0..inputs.len() {
                let output = self.series.forward(inputs[x].clone());
                if e % 100 == 0 {
                    let cost = self.loss.forward(output.clone(), targets[x].clone());
                    println!("Epoch: {}, Cost: {}", e, cost);
                }
                self.series.backward(CostObject::Loss(self.loss), self.lr);
            }
        }
    }
}
