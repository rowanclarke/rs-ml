use super::super::loss::Loss;
use super::{Cost, CostObject, Dynamic, Group, Layer, Object};
use ndarray::Array2;

pub struct Series {
    list: Vec<Object>,
    size: Vec<usize>,
}

impl Dynamic for Series {
    fn new(before: Vec<usize>) -> Self {
        Self {
            list: Vec::new(),
            size: before,
        }
    }
}

impl Group for Series {
    fn push(&mut self, object: Object) {
        self.size = object.after();
        self.list.push(object);
    }
}

impl Cost for Series {
    fn cost(&self, given: Array2<f32>) -> Array2<f32> {
        self.list[0].cost(given)
    }
}

impl Layer for Series {
    fn before(&self) -> Vec<usize> {
        self.size.clone()
    }

    fn after(&self) -> Vec<usize> {
        self.size.clone()
    }

    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        let mut buffer = input;
        for i in 0..self.list.len() {
            buffer = self.list[i].forward(buffer);
        }
        buffer
    }

    fn backward(&mut self, dele: CostObject, lr: f32) {
        for i in 0..self.list.len() {
            self.list[i].backward(buffer, lr);
        }
    }
}
