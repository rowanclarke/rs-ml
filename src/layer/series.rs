use super::{Dynamic, Group, Layer, Object};

pub struct Series {
    list: Vec<Object>,
    size: usize,
}

impl Dynamic for Series {
    fn new(before: usize) -> Self {
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
    fn before(&self) -> usize {
        self.size
    }

    fn after(&self) -> usize {
        self.size
    }

    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut buffer = input;
        for i in 0..self.list.len() {
            buffer = self.list[i].forward(buffer);
        }
        buffer
    }

    fn backward(&mut self) {}
}
