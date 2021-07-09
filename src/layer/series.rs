use super::Layer;

pub struct Series {
    list: Vec<Box<dyn Layer>>,
}

impl Series {
    pub fn new() -> Self {
        Self { list: Vec::new() }
    }

    pub fn push(&mut self, layer: Box<dyn Layer>) {
        self.list.push(layer);
    }
}

impl Layer for Series {
    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut buffer = input;
        for i in 0..self.list.len() {
            buffer = self.list[i].forward(buffer);
        }
        buffer
    }

    fn backward(&mut self) {}
}
