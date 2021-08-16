use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops;
use std::slice::Chunks;

pub type Array2 = Array<[usize; 2]>;
pub type Array3 = Array<[usize; 3]>;
pub type Array4 = Array<[usize; 4]>;
pub type Array5 = Array<[usize; 5]>;

pub trait Shape: ops::Index<usize, Output = usize> + IntoIterator<Item = usize> + Clone {}
impl<T> Shape for T where T: ops::Index<usize, Output = usize> + IntoIterator<Item = usize> + Clone {}

#[derive(Serialize, Deserialize, Clone)]
pub struct Array<T: Shape> {
    array: Vec<f32>,
    shape: T,
}

impl<T: Shape> Array<T> {
    pub fn random(shape: T) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            array: vec![0.0; shape.clone().into_iter().product()]
                .into_iter()
                .map(|_| rng.gen::<f32>())
                .collect(),
            shape,
        }
    }

    pub fn zeros(shape: T) -> Self {
        Self {
            array: vec![0.0; shape.clone().into_iter().product()],
            shape,
        }
    }

    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn from_shape_vec(shape: T, array: Vec<f32>) -> Array<T> {
        Array::<T> { array, shape }
    }

    pub fn into_raw_vec(self) -> Vec<f32> {
        self.array
    }

    pub fn shape(&self) -> T {
        self.shape.clone()
    }

    pub fn axis(&self) -> Chunks<'_, f32> {
        let mut iter_shape = self.shape.clone().into_iter();
        iter_shape.next();
        self.array.chunks(iter_shape.product())
    }
}

impl<'a, 'b, T: Shape> ops::Add<&'b Array<T>> for &'a Array<T> {
    type Output = Array<T>;

    fn add(self, rhs: &'b Array<T>) -> Array<T> {
        let mut result = Array::<T>::zeros(self.shape());
        for i in 0..result.len() {
            result.array[i] = self.array[i] + rhs.array[i];
        }
        result
    }
}

impl<'a, 'b, T: Shape> ops::Sub<&'b Array<T>> for &'a Array<T> {
    type Output = Array<T>;

    fn sub(self, rhs: &'b Array<T>) -> Array<T> {
        let mut result = Array::<T>::zeros(self.shape());
        for i in 0..result.len() {
            result.array[i] = self.array[i] - rhs.array[i];
        }
        result
    }
}

impl<'a, 'b, T: Shape> ops::Mul<&'b Array<T>> for &'a Array<T> {
    type Output = Array<T>;

    fn mul(self, rhs: &'b Array<T>) -> Array<T> {
        let mut result = Array::<T>::zeros(self.shape());
        for i in 0..result.len() {
            result.array[i] = self.array[i] * rhs.array[i];
        }
        result
    }
}

impl<'a, 'b, T: Shape> ops::Mul<f32> for &'a Array<T> {
    type Output = Array<T>;

    fn mul(self, rhs: f32) -> Array<T> {
        let mut result = Array::<T>::zeros(self.shape());
        for i in 0..result.len() {
            result.array[i] = self.array[i] * rhs;
        }
        result
    }
}

impl<T: Shape> ops::Index<T> for Array<T> {
    type Output = f32;

    fn index(&self, idx: T) -> &f32 {
        let mut iter_idx = idx.into_iter();
        let mut iter_shape = self.shape.clone().into_iter();
        let mut a_idx: usize = iter_idx.next().unwrap();
        iter_shape.next();
        for i in iter_idx {
            a_idx *= iter_shape.next().unwrap();
            a_idx += i;
        }
        &self.array[a_idx]
    }
}

impl<T: Shape> ops::IndexMut<T> for Array<T> {
    fn index_mut(&mut self, idx: T) -> &mut f32 {
        let mut iter_idx = idx.into_iter();
        let mut iter_shape = self.shape.clone().into_iter();
        let mut a_idx: usize = iter_idx.next().unwrap();
        iter_shape.next();
        for i in iter_idx {
            a_idx *= iter_shape.next().unwrap();
            a_idx += i;
        }
        &mut self.array[a_idx]
    }
}
