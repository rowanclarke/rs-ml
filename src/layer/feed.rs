use super::super::activation::Activation;
use super::{Layer, Template};
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
            bias: Array1::<f32>::zeros(self.after).map(|_| rng.gen::<f32>()),
            before: before[0],
            after: self.after,
            input: Vec::new(),
            sum: Vec::new(),
            output: Vec::new(),
            phantom: PhantomData,
        }
    }
}

pub struct FeedLayer<A: Activation> {
    weights: Array2<f32>,
    bias: Array1<f32>,
    before: usize,
    after: usize,
    input: Vec<f32>,
    sum: Vec<f32>,
    output: Vec<f32>,
    phantom: PhantomData<A>,
}

impl<A: Activation> Layer for FeedLayer<A> {
    fn before(&self) -> Vec<usize> {
        vec![self.before]
    }

    fn after(&self) -> Vec<usize> {
        vec![self.after]
    }

    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        self.input = input;
        self.sum = vec![0.0; self.after];
        for i in 0..self.after {
            self.sum[i] += self.bias[i];
            for j in 0..self.before {
                self.sum[i] += self.weights[[j, i]] * self.input[j];
            }
        }
        self.output = A::activate(self.sum.clone());
        self.output.clone()
    }

    fn backward(&mut self, target: Vec<f32>, lr: f32) -> Vec<f32> {
        let mut result = self.input.clone();
        let del = A::deactivate(self.sum.clone());
        for i in 0..self.after {
            let buffer = lr * 2.0 * (self.output[i] - target[i]) * del[i];
            self.bias[i] -= buffer;
            for j in 0..self.before {
                result[j] -= buffer * self.weights[[j, i]];
                self.weights[[j, i]] -= buffer * self.input[j];
            }
        }
        result
    }
}
