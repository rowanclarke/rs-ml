use super::layer::{series::Series, Dynamic, Fixed, Group, Layer, Object};

pub struct Model {
    stack: Vec<Box<dyn Group>>,
}

impl Model {
    pub fn new(size: usize) -> Self {
        Self {
            stack: vec![Box::new(Series::new(size))],
        }
    }

    pub fn push_group<G: Group + Dynamic>(&mut self) {
        let size = self.stack[self.stack.len() - 1].after();
        self.stack.push(Box::new(G::new(size)));
    }

    pub fn pop_group(&mut self) {
        let group = self.stack.pop().unwrap();
        let last = self.stack.len() - 1;
        self.stack[last].push(Object::Group(group));
    }

    pub fn push_layer<L: Layer + Fixed>(&mut self, after: usize) {
        let last = self.stack.len() - 1;
        let before = self.stack[last].before();
        self.stack[last].push(Object::Layer(Box::new(L::new(before, after))));
    }
}
