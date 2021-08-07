use super::super::{
    activation::Activation,
    matrix::{Column, Matrix},
};
use super::{Layer, LayerBuilder};
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
    fn build(self, before: Vec<usize>) -> Box<dyn Layer> {
        Box::new(FeedLayer::<A> {
            weights: Matrix::random((self.after, before[0])),
            bias: Column::random(self.after),
            input: Column::zeros(before[0]),
            sum: Column::zeros(self.after),
            output: Column::zeros(self.after),
            before: before[0],
            after: self.after,
            phantom: PhantomData,
        })
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
    phantom: PhantomData<A>,
}

impl<A: Activation> Layer for FeedLayer<A> {
    fn before(&self) -> Vec<usize> {
        vec![self.before]
    }

    fn after(&self) -> Vec<usize> {
        vec![self.after]
    }

    fn forward(&mut self, input: Column) -> Column {
        self.input = input;
        self.sum = &(&self.weights * &self.input) + &self.bias;
        self.output = A::activate(self.sum.clone());
        self.output.clone()
    }

    fn backward(&mut self, dc_y: Column, lr: f32) -> Column {
        let da = A::deactivate(self.sum.clone());
        let dc_a = &da * &dc_y;
        let mut ds_w = Matrix::zeros((self.after * self.before, self.after));
        for i in 0..self.after {
            for j in 0..self.before {
                ds_w[(i * self.after + j, i)] = self.input[j];
            }
        }
        let mut ds_b = Matrix::zeros((self.after, self.after));
        for i in 0..self.after {
            ds_b[(i, i)] = 1.0;
        }
        let mut ds_x = self.weights.clone();
        ds_x.transpose();
        let mut lr_dc_w = (&(&ds_w * &dc_a) * lr).to_mat();
        lr_dc_w.reshape(self.weights.shape());
        self.weights -= &lr_dc_w;
        self.bias -= &(&(&ds_b * &dc_a) * lr);
        &ds_x * &dc_a
    }
}
