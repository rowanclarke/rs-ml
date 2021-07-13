use super::{Dynamic, Group, Layer, Object};

pub struct Parallel {
    list: Vec<Object>,
    before: usize,
    after: usize,
}

impl Dynamic for Parallel {
    fn new(before: usize) -> Self {
        Self {
            list: Vec::new(),
            before: before,
            after: 0,
        }
    }
}

impl Group for Parallel {
    fn push(&mut self, object: Object) {
        self.after += object.after();
        self.list.push(object);
    }
}

impl Layer for Parallel {
    fn before(&self) -> usize {
        self.before
    }

    fn after(&self) -> usize {
        self.after
    }

    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut buffer = Vec::<f32>::new();
        for i in 0..self.list.len() {
            buffer.append(self.list[i].forward(input.clone()).as_mut());
        }
        buffer
    }

    fn backward(&mut self, target: Vec<f32>, lr: f32) -> Vec<f32> {
        let mut buffer = vec![0.0; self.before];
        for i in self.list.len()..0 {
            let result = self.list[i - 1].backward(target[0..2].to_vec(), lr);
            for j in self.list.len()..0 {
                buffer[j] += result[j];
            }
        }
        buffer
    }
}
