use super::super::{
    activation::Activation,
    matrix::{Column, Jacobean, Matrix},
};
use super::{Cost, CostObject, Layer, LayerBuilder};
use ndarray::Ix2;
use rand::prelude::*;
use std::marker::PhantomData;

pub struct Feed<A: Activation> {
    after: usize,
    phantom: PhantomData<A>,
}

impl<A: Activation> Feed<A> {
    pub fn new(after: usize) -> Self {
        Self {
            after,
            phantom: PhantomData,
        }
    }
}

impl<A: Activation> LayerBuilder for Feed<A> {
    fn build(&self, before: Vec<usize>, other: CostObject) -> Box<dyn Layer> {
        Box::new(FeedLayer::<A> {
            weights: Matrix::random((self.after, before[0])),
            bias: Column::random(self.after),
            input: Column::zeros(before[0]),
            sum: Column::zeros(self.after),
            output: Column::zeros(self.after),
            before: before[0],
            after: self.after,
            other,
            phantom: PhantomData,
        })
    }

    fn after(&self, before: Vec<usize>) -> Vec<usize> {
        vec![self.after]
    }
}

pub struct FeedLayer<A: Activation> {
    pub weights: Matrix,
    pub bias: Column,
    input: Column,
    sum: Column,
    output: Column,
    before: usize,
    after: usize,
    other: CostObject,
    phantom: PhantomData<A>,
}

impl<A: Activation> FeedLayer<A> {
    pub fn ds_x(&self) -> Jacobean {
        self.weights.clone()
    }

    pub fn ds_w(&self) -> Jacobean {
        let mut ds_w = Jacobean::zeros((self.after, self.after * self.before));
        for i in 0..self.after {
            for j in 0..self.before {
                ds_w[(i, i * self.before + j)] = self.input[j];
            }
        }
        ds_w
    }

    pub fn ds_b(&self) -> Jacobean {
        let mut ds_b = Jacobean::zeros((self.after, self.after));
        for i in 0..self.after {
            ds_b[(i, i)] = 1.0;
        }
        ds_b
    }
}

impl<A: Activation> Cost for FeedLayer<A> {
    fn cost(&self, given: Jacobean) -> Jacobean {
        let da_s = A::deactivate(self.sum.clone());
        let ds_x = self.ds_x();
        let da_wb = &da_s * &(&ds_x * &given);
        self.other.cost(da_wb)
    }
}

impl<A: Activation> Layer for FeedLayer<A> {
    fn before(&self) -> Vec<usize> {
        vec![self.before]
    }

    fn after(&self) -> Vec<usize> {
        vec![self.after]
    }

    fn train(&mut self, input: Column, target: Column, lr: f32) {
        self.forward(input);
        match &mut self.other {
            CostObject::Layer(layer) => layer.train(self.output.clone(), target, lr),
            CostObject::Loss(loss) => loss.train(self.output.clone(), target),
        };
        self.backward(lr);
    }

    fn test(&mut self, input: Column) {
        self.forward(input);
        match &mut self.other {
            CostObject::Layer(layer) => layer.test(self.output.clone()),
            CostObject::Loss(_) => println!("{}", self.output),
        };
    }

    fn forward(&mut self, input: Column) {
        self.input = input;
        self.sum = &(&self.weights * &self.input) + &self.bias;
        self.output = A::activate(self.sum.clone());
    }

    fn backward(&mut self, lr: f32) {
        let da_s = A::deactivate(self.sum.clone());
        let da_w = &da_s * &self.ds_w();
        let da_b = &da_s * &self.ds_b();
        let dc_w = self.other.cost(da_w);
        let dc_b = self.other.cost(da_b);
        self.weights = &self.weights - lr * dc_w;
        self.bias = &self.bias - lr * dc_b;
    }
}
