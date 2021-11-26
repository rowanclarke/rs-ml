use super::super::{
    activation::{Activation, ActivationBuilder},
    array::{Array3, Array4},
    matrix::{Column, Jacobean},
};
use super::{Layer, LayerBuilder};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Serialize, Deserialize)]
pub struct Conv2DLayer {
    activation: Box<dyn Activation>,
    filter: Array4,
    input: Array3,
    sum: Array3,
    output: Array3,
    before: Vec<usize>,
    after: Vec<usize>,
}

#[typetag::serde]
impl Layer for Conv2DLayer {
    fn before(&self) -> Vec<usize> {
        self.before.clone()
    }

    fn after(&self) -> Vec<usize> {
        self.after.clone()
    }

    fn forward(&mut self, input: Column) -> Column {
        self.input = input
            .clone()
            .to_arr([self.before[0], self.before[1], self.before[2]]);
        let sum = &self.ds_x() * &input;
        self.sum = sum
            .clone()
            .to_arr([self.after[0], self.after[1], self.after[2]]);
        let output = self.activation.activate(sum);
        self.output = output
            .clone()
            .to_arr([self.after[0], self.after[1], self.after[2]]);
        output
    }

    fn backward(&mut self, dc_y: Column, lr: f32) -> Column {
        let sum = Column::from_arr(self.sum.clone());
        let mut dc_f = Array4::zeros(self.filter.shape());
        let dy_s = self.activation.deactivate(sum);
        let mut ds_x = self.ds_x();
        ds_x.transpose();
        let dc_s = &dy_s * &dc_y;
        let dc_x = &ds_x * &dc_s;
        let dc_s = dc_s.to_arr([self.after[0], self.after[1], self.after[2]]);
        Self::back_convolution(&self.input, &dc_s, &mut dc_f);
        self.filter = &self.filter - &(&dc_f * lr);
        dc_x
    }
}

#[derive(Serialize, Deserialize)]
pub struct MaxPooling2DLayer {
    pool_size: (usize, usize),
    input: Array3,
    output: Array3,
    before: Vec<usize>,
    after: Vec<usize>,
}

#[typetag::serde]
impl Layer for MaxPooling2DLayer {
    fn before(&self) -> Vec<usize> {
        self.before.clone()
    }

    fn after(&self) -> Vec<usize> {
        self.after.clone()
    }

    fn forward(&mut self, input: Column) -> Column {
        self.input = input.to_arr([self.before[0], self.before[1], self.before[2]]);
        for i in self.output.iter() {
            let mut max: f32 = 0.0;
            for x in 0..self.pool_size.0 {
                for y in 0..self.pool_size.1 {
                    max = max.max(
                        self.input[[
                            self.pool_size.0 * i[0] + x,
                            self.pool_size.1 * i[1] + y,
                            i[2],
                        ]],
                    )
                }
            }
            self.output[i] = max;
        }
        Column::from_arr(self.output.clone())
    }

    fn backward(&mut self, dc_y: Column, _: f32) -> Column {
        let dc_y = dc_y.to_arr([self.after[0], self.after[1], self.after[2]]);
        for i in self.input.iter() {
            if self.input[i]
                == self.output[[i[0] / self.pool_size.0, i[1] / self.pool_size.1, i[2]]]
            {
                self.input[i] = dc_y[[i[0] / self.pool_size.0, i[1] / self.pool_size.1, i[2]]];
            }
        }
        Column::from_arr(self.input.clone())
    }
}

pub struct MaxPooling2D {
    pool_size: (usize, usize),
}

impl MaxPooling2D {
    pub fn new(pool_size: (usize, usize)) -> Self {
        Self { pool_size }
    }
}

impl LayerBuilder for MaxPooling2D {
    fn build(self, before: Vec<usize>) -> Box<dyn Layer> {
        let after = vec![
            before[0] / self.pool_size.0,
            before[1] / self.pool_size.1,
            before[2],
        ];
        Box::new(MaxPooling2DLayer {
            pool_size: self.pool_size,
            input: Array3::zeros([before[0], before[1], before[2]]),
            output: Array3::zeros([after[0], after[1], after[2]]),
            before,
            after,
        })
    }
}

pub struct Conv2D<A: Activation> {
    activation: A,
    filters: usize,
    kernel_size: (usize, usize),
}

impl<A: Activation + ActivationBuilder> Conv2D<A> {
    pub fn new(filters: usize, kernel_size: (usize, usize)) -> Self {
        Self {
            activation: A::new(),
            filters,
            kernel_size,
        }
    }
}

impl<A: Activation> LayerBuilder for Conv2D<A> {
    fn build(self, before: Vec<usize>) -> Box<dyn Layer> {
        let mut rng = rand::thread_rng();
        let after = vec![
            before[0] - self.kernel_size.0 + 1,
            before[1] - self.kernel_size.1 + 1,
            self.filters,
        ];
        Box::new(Conv2DLayer {
            activation: Box::new(self.activation),
            filter: Array4::random([
                self.kernel_size.0,
                self.kernel_size.1,
                before[2],
                self.filters,
            ]),
            input: Array3::zeros([before[0], before[1], before[2]]),
            sum: Array3::zeros([after[0], after[1], after[2]]),
            output: Array3::zeros([after[0], after[1], after[2]]),
            before,
            after,
        })
    }
}

impl Conv2DLayer {
    pub fn ds_x(&self) -> Jacobean {
        let mut ds_x = Jacobean::zeros((self.output.len(), self.input.len()));
        let li = self.input.shape();
        let lo = self.output.shape();
        let lf = self.filter.shape();
        for i in self.output.iter_filter(|x| x[2] > 0) {
            for j in self.filter.iter() {
                ds_x[(
                    lo[2] * (lo[1] * i[0] + i[1]) + j[3],
                    lf[2] * (li[1] * (j[0] + i[0]) + j[1] + i[1]) + j[2],
                )] = self.filter[j];
            }
        }
        ds_x
    }

    pub fn back_convolution(input: &Array3, output: &Array3, filter: &mut Array4) {
        for i in filter.iter() {
            let mut sum = 0.0;
            for j in output.iter_filter(|x| x[2] > 0) {
                sum += input[[i[0] + j[0], i[1] + j[1], i[2]]] * output[[j[0], j[1], i[3]]];
            }
            filter[i] = sum;
        }
    }
}
