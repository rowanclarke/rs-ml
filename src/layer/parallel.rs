use super::Layer;

pub struct Parallel {
    list: Vec<Box<dyn Layer>>,
}

impl Parallel {
    pub fn new() -> Self {
        Self { list: Vec::new() }
    }

    pub fn push(&mut self, layer: Box<dyn Layer>) {
        self.list.push(layer);
    }
}

impl Layer for Parallel {
    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut buffer = Vec::<f32>::new();
        for i in 0..self.list.len() {
            buffer.append(self.list[i].forward(input.clone()).as_mut());
        }
        buffer
    }

    fn backward(&mut self) {}
}
