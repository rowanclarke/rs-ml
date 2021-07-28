use super::super::activation::Activation;
use super::super::loss::Loss;
use super::{Cost, CostObject, Layer, Template};
use ndarray::{Array1, Array2};
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

impl<A: Activation> Template<FeedLayer<A>> for Feed<A> {
    fn into(self, before: Vec<usize>) -> FeedLayer<A> {
        let mut rng = rand::thread_rng();
        FeedLayer::<A> {
            weights: Array2::<f32>::zeros((before[0], self.after)).map(|_| rng.gen::<f32>()),
            bias: Array2::<f32>::zeros((1, self.after)).map(|_| rng.gen::<f32>()),
            input: Array2::<f32>::zeros((1, before[0])),
            sum: Array2::<f32>::zeros((1, self.after)),
            output: Array2::<f32>::zeros((1, self.after)),
            before: before[0],
            after: self.after,
            phantom: PhantomData,
        }
    }
}

pub struct FeedLayer<A: Activation> {
    weights: Array2<f32>,
    bias: Array2<f32>,
    input: Array2<f32>,
    sum: Array2<f32>,
    output: Array2<f32>,
    before: usize,
    after: usize,
    phantom: PhantomData<A>,
}

impl<A: Activation> FeedLayer<A> {
    pub fn dels_x(&self) -> Array2<f32> {
        self.weights
    }

    pub fn dels_w(&self) -> Array2<f32> {
        let del = Array2::zeros((self.after, self.after * self.before));
        for i in 0..self.after {
            for j in 0..self.before {
                del[[i, i * self.before + j]] = self.input[[j, 0]];
            }
        }
        del
    }

    pub fn dels_b(&self) -> Array2<f32> {
        let del = Array2::zeros((self.after, self.after));
        for i in 0..self.after {
            del[[i, i]] = 1.0;
        }
        del
    }
}

impl<A: Activation> Cost for FeedLayer<A> {
    fn cost(&self, given: Array2<f32>) -> Array2<f32> {}
}

impl<A: Activation> Layer for FeedLayer<A> {
    fn before(&self) -> Vec<usize> {
        vec![self.before]
    }

    fn after(&self) -> Vec<usize> {
        vec![self.after]
    }

    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        self.input = input;
        self.sum = self.weights.dot(&self.input) + self.bias;
        self.output = A::activate(self.sum.clone());
        self.output.clone()
    }

    fn backward(&mut self, dele: CostObject, lr: f32) {
        let dela = A::deactivate(self.sum.clone());
        let delw = dele
            .cost(dela.dot(&self.dels_w()))
            .into_shape(self.weights.shape())
            .unwrap();
        let delb = dele
            .cost(dela.dot(&self.dels_b()))
            .into_shape(self.bias.shape())
            .unwrap();
        self.weights = self.weights - lr * delw;
        self.bias = self.bias - lr * delb;
    }
}
