use super::super::{
    activation::Activation,
    matrix::{Column, Jacobean},
};
use super::{Layer, LayerBuilder};
use ndarray::{Array3, Array4};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

pub struct Conv2D<A: Activation> {
    filters: usize,
    kernel_size: (usize, usize),
    phantom: PhantomData<A>,
}

impl<A: Activation> Conv2D<A> {
    pub fn new(filters: usize, kernel_size: (usize, usize)) -> Self {
        Self {
            filters,
            kernel_size,
            phantom: PhantomData,
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
        Box::new(Conv2DLayer::<A> {
            filter: Array4::<f32>::zeros((
                self.kernel_size.0,
                self.kernel_size.1,
                before[2],
                self.filters,
            ))
            .map(|_| rng.gen::<f32>()),
            input: Array3::<f32>::zeros((before[0], before[1], before[2])),
            sum: Array3::<f32>::zeros((after[0], after[1], after[2])),
            output: Array3::<f32>::zeros((after[0], after[1], after[2])),
            before,
            after,
            phantom: PhantomData,
        })
    }
}

#[derive(Serialize, Deserialize)]
pub struct Conv2DLayer<A: Activation> {
    filter: Array4<f32>,
    input: Array3<f32>,
    sum: Array3<f32>,
    output: Array3<f32>,
    before: Vec<usize>,
    after: Vec<usize>,
    phantom: PhantomData<A>,
}

impl<A: Activation> Layer for Conv2DLayer<A> {
    fn before(&self) -> Vec<usize> {
        self.before.clone()
    }

    fn after(&self) -> Vec<usize> {
        self.after.clone()
    }

    fn forward(&mut self, input: Column) -> Column {
        self.input = input
            .clone()
            .to_arr((self.before[0], self.before[1], self.before[2]));
        Self::convolution(&self.input, &self.filter, &mut self.sum);
        let sum = Column::from_arr(self.sum.clone());
        //asserteq!(sum, (&self.ds_x() * &input));
        let output = A::activate(sum);
        self.output = output
            .clone()
            .to_arr((self.after[0], self.after[1], self.after[2]));
        output
    }

    fn backward(&mut self, dc_y: Column, lr: f32) -> Column {
        let sum = Column::from_arr(self.sum.clone());
        let mut dc_f = Array4::<f32>::zeros(self.filter.raw_dim());
        let dy_s = A::deactivate(sum);
        let mut ds_x = self.ds_x();
        ds_x.transpose();
        let dc_s = &dy_s * &dc_y;
        let dc_x = &ds_x * &dc_s;
        let dc_s = dc_s.to_arr((self.after[0], self.after[1], self.after[2]));
        Self::back_convolution(&self.input, &dc_s, &mut dc_f);
        self.filter = &self.filter - &(lr * dc_f);
        dc_x
    }
}

impl<A: Activation> Conv2DLayer<A> {
    pub fn ds_x(&self) -> Jacobean {
        let mut ds_x = Jacobean::zeros((self.output.len(), self.input.len()));
        let li = self.input.shape();
        let lo = self.output.shape();
        let lf = self.filter.shape();
        for i in 0..lo[0] {
            for j in 0..lo[1] {
                for k in 0..lo[2] {
                    for x in 0..lf[0] {
                        for y in 0..lf[1] {
                            for z in 0..lf[2] {
                                ds_x[(
                                    lo[2] * (lo[1] * i + j) + k,
                                    lf[2] * (li[1] * (x + i) + y + j) + z,
                                )] = self.filter[[x, y, z, k]];
                            }
                        }
                    }
                }
            }
        }
        ds_x
    }

    pub fn convolution(input: &Array3<f32>, filter: &Array4<f32>, output: &mut Array3<f32>) {
        for i in 0..output.shape()[0] {
            for j in 0..output.shape()[1] {
                for k in 0..output.shape()[2] {
                    let mut sum = 0.0;
                    for x in 0..filter.shape()[0] {
                        for y in 0..filter.shape()[1] {
                            for z in 0..filter.shape()[2] {
                                sum += input[[i + x, j + y, z]] * filter[[x, y, z, k]];
                            }
                        }
                    }
                    output[[i, j, k]] = sum;
                }
            }
        }
    }

    pub fn back_convolution(input: &Array3<f32>, output: &Array3<f32>, filter: &mut Array4<f32>) {
        for i in 0..filter.shape()[0] {
            for j in 0..filter.shape()[1] {
                for k in 0..filter.shape()[2] {
                    for l in 0..filter.shape()[3] {
                        let mut sum = 0.0;
                        for x in 0..output.shape()[0] {
                            for y in 0..output.shape()[1] {
                                sum += input[[i + x, j + y, k]] * output[[x, y, l]];
                            }
                        }
                        filter[[i, j, k, l]] = sum;
                    }
                }
            }
        }
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
            input: Array3::<f32>::zeros((before[0], before[1], before[2])),
            output: Array3::<f32>::zeros((after[0], after[1], after[2])),
            before,
            after,
        })
    }
}

#[derive(Serialize, Deserialize)]
pub struct MaxPooling2DLayer {
    pool_size: (usize, usize),
    input: Array3<f32>,
    output: Array3<f32>,
    before: Vec<usize>,
    after: Vec<usize>,
}

impl Layer for MaxPooling2DLayer {
    fn before(&self) -> Vec<usize> {
        self.before.clone()
    }

    fn after(&self) -> Vec<usize> {
        self.after.clone()
    }

    fn forward(&mut self, input: Column) -> Column {
        self.input = input.to_arr((self.before[0], self.before[1], self.before[2]));
        for i in 0..self.after[0] {
            for j in 0..self.after[1] {
                for k in 0..self.after[2] {
                    let mut max: f32 = 0.0;
                    for x in 0..self.pool_size.0 {
                        for y in 0..self.pool_size.1 {
                            max = max.max(
                                self.input[[self.pool_size.0 * i + x, self.pool_size.1 * j + y, k]],
                            )
                        }
                    }
                    self.output[[i, j, k]] = max;
                }
            }
        }
        Column::from_arr(self.output.clone())
    }

    fn backward(&mut self, dc_y: Column, _: f32) -> Column {
        let dc_y = dc_y.to_arr((self.after[0], self.after[1], self.after[2]));
        for i in 0..self.before[0] {
            for j in 0..self.before[1] {
                for k in 0..self.before[2] {
                    if self.input[[i, j, k]]
                        == self.output[[i / self.pool_size.0, j / self.pool_size.1, k]]
                    {
                        self.input[[i, j, k]] =
                            dc_y[[i / self.pool_size.0, j / self.pool_size.1, k]];
                    }
                }
            }
        }
        Column::from_arr(self.input.clone())
    }
}
