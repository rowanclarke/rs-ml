use super::super::{
    activation::{Activation, ActivationBuilder},
    matrix::{Column, Matrix},
};
use super::{Layer, LayerBuilder};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Serialize, Deserialize)]
pub struct FeedLayer {
    activation: Box<dyn Activation>,
    weights: Matrix,
    bias: Column,
    input: Column,
    sum: Column,
    output: Column,
    before: usize,
    after: usize,
}

#[typetag::serde]
impl Layer for FeedLayer {
    fn before(&self) -> Vec<usize> {
        vec![self.before]
    }

    fn after(&self) -> Vec<usize> {
        vec![self.after]
    }

    fn forward(&mut self, input: Column) -> Column {
        self.input = input;
        self.sum = &(&self.weights * &self.input) + &self.bias;
        self.output = self.activation.activate(self.sum.clone());
        self.output.clone()
    }

    fn backward(&mut self, dc_y: Column, lr: f32) -> Column {
        let dy_s = self.activation.deactivate(self.sum.clone());
        let dc_s = &dy_s * &dc_y;
        let mut ds_w = Matrix::zeros((self.after * self.before, self.after));
        for i in 0..self.after {
            for j in 0..self.before {
                ds_w[(i * self.before + j, i)] = self.input[j];
            }
        }
        let mut ds_b = Matrix::zeros((self.after, self.after));
        for i in 0..self.after {
            ds_b[(i, i)] = 1.0;
        }
        let mut ds_x = self.weights.clone();
        ds_x.transpose();
        let dc_w = &ds_w * &dc_s;
        let mut lr_dc_w = (&dc_w * lr).to_mat();
        lr_dc_w.reshape(self.weights.shape());
        self.weights -= &lr_dc_w;
        self.bias -= &(&(&ds_b * &dc_s) * lr);
        &ds_x * &dc_s
    }
}

pub struct Feed<A: Activation> {
    activation: A,
    after: usize,
}

impl<A: Activation + ActivationBuilder> Feed<A> {
    pub fn new(after: usize) -> Self {
        Self {
            activation: A::new(),
            after,
        }
    }
}

impl<A: Activation> LayerBuilder for Feed<A> {
    fn build(self, before: Vec<usize>) -> Box<dyn Layer> {
        Box::new(FeedLayer {
            activation: Box::new(self.activation),
            weights: Matrix::random((self.after, before[0])),
            bias: Column::random(self.after),
            input: Column::zeros(before[0]),
            sum: Column::zeros(self.after),
            output: Column::zeros(self.after),
            before: before[0],
            after: self.after,
        })
    }
}
