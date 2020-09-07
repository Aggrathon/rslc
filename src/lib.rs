
mod utils;

use ndarray::{s, Array1, Array2, indices_of};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::RuntimeError;
use pyo3::prelude::{pymodule, Py, PyErr, PyModule, PyResult, Python};
use std::fmt::Display;
use std::collections::VecDeque;

use utils::Combinations;


fn make_error<E: Display + Sized>(e: E) -> PyErr {
    PyErr::new::<RuntimeError, _>(format!("[rslc] {}", e))
}

pub trait DistanceMeasure {
    fn is_finite(&self) -> bool;
    const INFINITE_VALUE: Self;
    const MINIMUM_VALUE: Self;
}

impl DistanceMeasure for f32 {
    fn is_finite(&self) -> bool { *self >= 0.0 }
    const INFINITE_VALUE: Self = -1.0;
    const MINIMUM_VALUE: Self = 0.0;
}

impl DistanceMeasure for f64 {
    fn is_finite(&self) -> bool { *self >= 0.0 }
    const INFINITE_VALUE: Self = -1.0;
    const MINIMUM_VALUE: Self = 0.0;
}

impl DistanceMeasure for isize {
    fn is_finite(&self) -> bool { *self >= 0 }
    const INFINITE_VALUE: Self = -1;
    const MINIMUM_VALUE: Self = 0;
}

impl DistanceMeasure for i16 {
    fn is_finite(&self) -> bool { *self >= 0 }
    const INFINITE_VALUE: Self = -1;
    const MINIMUM_VALUE: Self = 0;
}

impl DistanceMeasure for i32 {
    fn is_finite(&self) -> bool { *self >= 0 }
    const INFINITE_VALUE: Self = -1;
    const MINIMUM_VALUE: Self = 0;
}

impl DistanceMeasure for i64 {
    fn is_finite(&self) -> bool { *self >= 0 }
    const INFINITE_VALUE: Self = -1;
    const MINIMUM_VALUE: Self = 0;
}

impl DistanceMeasure for i128 {
    fn is_finite(&self) -> bool { *self >= 0 }
    const INFINITE_VALUE: Self = -1;
    const MINIMUM_VALUE: Self = 0;
}

impl DistanceMeasure for usize {
    fn is_finite(&self) -> bool { *self != Self::MAX }
    const INFINITE_VALUE: Self = Self::MAX;
    const MINIMUM_VALUE: Self = 0;
}

impl DistanceMeasure for u16 {
    fn is_finite(&self) -> bool { *self != Self::MAX }
    const INFINITE_VALUE: Self = Self::MAX;
    const MINIMUM_VALUE: Self = 0;
}

impl DistanceMeasure for u32 {
    fn is_finite(&self) -> bool { *self != Self::MAX }
    const INFINITE_VALUE: Self = Self::MAX;
    const MINIMUM_VALUE: Self = 0;
}

impl DistanceMeasure for u64 {
    fn is_finite(&self) -> bool { *self != Self::MAX }
    const INFINITE_VALUE: Self = Self::MAX;
    const MINIMUM_VALUE: Self = 0;
}

impl DistanceMeasure for u128 {
    fn is_finite(&self) -> bool { *self != Self::MAX }
    const INFINITE_VALUE: Self = Self::MAX;
    const MINIMUM_VALUE: Self = 0;
}

struct FloodFill {
    start: usize,
    current: usize,
    cache: VecDeque<usize>,
    clusters: Array1<usize>,
    sizes: Vec<(usize, usize)>,
}

impl FloodFill {
    fn new() -> Self {
        FloodFill {
            start: 0,
            current: 0,
            cache: VecDeque::new(),
            clusters: Array1::zeros(0),
            sizes: vec!(),
        }
    }

    fn flood_fill<A>(&mut self, distances: &Array2<A>) where A: DistanceMeasure {
        if self.clusters.len() != distances.ncols() {
            self.clusters = Array1::zeros(distances.ncols());
        }
        self.start = self.current + 1;
        let mut current = self.start;
        self.sizes.clear();
        while let Some((i, _)) = self.clusters.iter().enumerate().find(|(_, x)| **x < self.start) {
            self.cache.clear();
            self.clusters[i] = current;
            self.cache.push_back(i);
            let mut count = 1;
            while let Some(i) = self.cache.pop_front() {
                for (j, _) in distances.outer_iter().nth(i).unwrap().iter().enumerate().filter(|(_, y)| A::is_finite(*y)) {
                    if self.clusters[j] != current {
                        self.clusters[j] = current;
                        self.cache.push_back(j);
                        count += 1;
                    }
                }
            }
            self.sizes.push((current, count));
            current += 1;
        }
        self.current = current;
    }
}

fn find_outliers(flood_fill: &FloodFill, outliers: &mut Array1<bool>, min_size: usize) -> bool {
    let mut new_outliers = false;
    for (i, _) in flood_fill.sizes.iter().filter(|(_, s)| *s < min_size) {
        for (o, _) in outliers.iter_mut().zip(flood_fill.clusters.iter()).filter(|(_, c)| *c == i) {
            *o = true;
        }
        new_outliers = true;
    }
    new_outliers
}

fn rslc_slow<A>(distances: &Array2<A>, clusters: usize, min_size: usize) -> (Array1<usize>, Array1<bool>) where A: DistanceMeasure + Copy + PartialOrd {
    let mut dists = distances.to_owned();
    let mut outliers = Array1::default(distances.ncols());
    let mut flood_fill = FloodFill::new();
    let mut order: Vec<(usize, usize)> = Combinations::iter(dists.ncols()).collect();
    order.sort_unstable_by(|a, b| dists[*a].partial_cmp(&dists[*b]).unwrap());
    for (i, j) in order.into_iter() {
        let d = dists[[i, j]];
        dists[[i, j]] = A::INFINITE_VALUE;
        dists[[j, i]] = A::INFINITE_VALUE;
        flood_fill.flood_fill(&dists);
        if flood_fill.sizes.len() > clusters {
            dists[[i, j]] = d;
            dists[[j, i]] = d;
            flood_fill.flood_fill(&dists);
            break;
        }
        if find_outliers(&flood_fill, &mut outliers, min_size) {
            dists[[i, j]] = A::MINIMUM_VALUE;
        }
    }
    (flood_fill.clusters - flood_fill.start, outliers)
}


fn rslc() {
    //TODO order dists descending
    // loop
        // remove n dists
        // flood_fill
        // check number of clusters
            // backtrack until correct number
        // check outliers
            // backtrack until no outliers, mark outliers, change dist to 0
        // if correct number of clusters and no outliers
            // break
    // return clusters and outliers
}



#[pymodule]
fn rust_linalg(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    
    #[pyfn(m, "inv")]
    fn inv(py: Python<'_>, x: &PyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        // let x = x.as_array().inv().map_err(make_error)?;
        // Ok(x.into_pyarray(py).to_owned())
        Ok(x.to_owned())
    }

    Ok(())
}




#[cfg(test)]
mod tests {
    use ndarray::{Array1, array, Array2};
    use super::*;

    #[test]
    fn rslc1() {
        let x = array![[0,4,3],[4,0,2],[3,2,0]];
        rslc_slow(&x, 3, 0);
    }
}
