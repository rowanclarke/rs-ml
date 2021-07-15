use super::layer::{series::Series, Dynamic, Fixed, Group, Layer, Object};
use std::mem;

pub struct Model {
    stack: Vec<Box<dyn Group>>,
    series: Box<Series>,
}

impl Model {
    pub fn new(size: Vec<usize>) -> Self {
        Self {
            stack: Vec::new(),
            series: Box::new(Series::new(size)),
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

    pub fn push_layer<L: Layer + Fixed>(&mut self, after: Vec<usize>) {
        let last = self.last();
        let before = last.before();
        last.push(Object::Layer(Box::new(L::new(before, after))));
    }

    pub fn train(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], epochs: u32, lr: f32) {
        for e in 0..epochs {
            for x in 0..inputs.len() {
                let output = self.series.forward(inputs[x].clone());
                if (e % 500) == 0 {
                    let mut cost = 0.0;
                    for i in 0..self.series.after()[0] {
                        cost += (output[i] - targets[x][i]).powf(2.0);
                    }
                    println!(
                        "Epoch: {}, Input: {:?}, Output: {:?}, Cost: {}",
                        e, inputs[x], output, cost
                    );
                }
                self.series.backward(targets[x].clone(), lr);
            }
        }
    }
}
