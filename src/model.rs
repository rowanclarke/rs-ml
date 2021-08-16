use super::layer::{Layer, LayerBuilder};
use super::loss::{Loss, LossBuilder};
use super::matrix::Column;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

static BASE_PATH: &str = "model/";

pub struct ModelBuilder<'a> {
    name: &'a str,
    layers: Vec<Box<dyn Layer>>,
    before: Vec<usize>,
}

impl<'a> ModelBuilder<'a> {
    pub fn new(name: &'a str, before: Vec<usize>) -> Self {
        Self {
            name,
            layers: vec![],
            before,
        }
    }

    pub fn push_layer<L: LayerBuilder>(&mut self, template: L) {
        let layer = template.build(self.before.clone());
        self.before = layer.after();
        self.layers.push(layer);
    }

    pub fn compile<L: Loss + LossBuilder>(self, lr: f32) -> Model<'a> {
        Model {
            name: self.name,
            lr,
            loss: Box::new(L::new()),
            layers: self.layers,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Model<'a> {
    name: &'a str,
    lr: f32,
    loss: Box<dyn Loss>,
    layers: Vec<Box<dyn Layer>>,
}

impl<'a> Model<'a> {
    pub fn train(&mut self, inputs: Box<[Column]>, targets: Box<[Column]>, epochs: u32) {
        println!("TRAINING");
        for e in 0..epochs {
            for i in 0..inputs.len() {
                let cost = self.train_one(inputs[i].clone(), targets[i].clone());
                println!("{}: cost: {}", e, cost);
            }
        }
    }

    pub fn validate(&mut self, inputs: Box<[Column]>, targets: Box<[Column]>) {
        println!("VALIDATING");
        for i in 0..inputs.len() {
            let cost = self.validate_one(inputs[i].clone(), targets[i].clone());
            println!("cost: {}", cost);
        }
    }

    pub fn test(&mut self, inputs: Box<[Column]>) {
        println!("TESTING");
        for i in 0..inputs.len() {
            let y = self.test_one(inputs[i].clone());
            println!("{}", y);
        }
    }

    pub fn train_one(&mut self, input: Column, target: Column) -> f32 {
        let y = self.test_one(input);
        let mut dc_y = self.loss.backward(y.clone(), target.clone());
        for l in (0..self.layers.len()).rev() {
            dc_y = self.layers[l].backward(dc_y, self.lr);
        }
        self.loss.forward(y, target)
    }

    pub fn validate_one(&mut self, input: Column, target: Column) -> f32 {
        let y = self.test_one(input);
        self.loss.forward(y, target)
    }

    pub fn test_one(&mut self, input: Column) -> Column {
        let mut x = input;
        for l in 0..self.layers.len() {
            x = self.layers[l].forward(x);
        }
        x
    }
}

pub struct ModelFiler<'a> {
    base_path: &'a str,
}

impl<'a> ModelFiler<'a> {
    pub fn new() -> Self {
        Self {
            base_path: BASE_PATH,
        }
    }

    pub fn base_path(&mut self, base_path: &'a str) -> &mut Self {
        self.base_path = base_path;
        self
    }

    pub fn save(&self, model: &Model) {
        let bin = bincode::serialize(model).unwrap();
        let path = &Path::new(self.base_path).join(model.name);
        let mut fh = OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap_or_else(|_| panic!("Unable to find path to model at {:?}.", path));
        fh.write_all(bin.as_slice()).unwrap();
    }

    pub fn load(&self, name: &'a str) -> Model {
        let mut buffer = Vec::<u8>::new();
        let path = &Path::new(self.base_path).join(name);
        let mut fh = OpenOptions::new()
            .read(true)
            .open(path)
            .unwrap_or_else(|_| panic!("Unable to find path to model at {:?}.", path));
        let _ = fh.read_to_end(&mut buffer).unwrap();
        bincode::deserialize::<Model>(buffer.leak()).unwrap()
    }
}
