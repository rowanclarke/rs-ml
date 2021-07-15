use super::{Dynamic, Group, Layer, Object};

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

impl Layer for Series {
    fn before(&self) -> Vec<usize> {
        self.size.clone()
    }

    fn after(&self) -> Vec<usize> {
        self.size.clone()
    }

    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut buffer = input;
        for i in 0..self.list.len() {
            buffer = self.list[i].forward(buffer);
        }
        buffer
    }

    fn backward(&mut self, target: Vec<f32>, lr: f32) -> Vec<f32> {
        let mut buffer = target;
        for i in (0..self.list.len()).rev() {
            buffer = self.list[i].backward(buffer, lr);
        }
        buffer
    }
}
