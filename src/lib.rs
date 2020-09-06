use ndarray::{s, Array1, Array2};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::RuntimeError;
use pyo3::prelude::{pymodule, Py, PyErr, PyModule, PyResult, Python};
use std::fmt::Display;
use std::collections::VecDeque;


fn make_error<E: Display + Sized>(e: E) -> PyErr {
    PyErr::new::<RuntimeError, _>(format!("[rslc] {}", e))
}

fn linear_index(i: usize, j: usize, w: usize) -> usize {
    assert!(i != j);
    assert!(j < w);
    assert!(i < w);
    if i < j { 
        _linear_index(j, i, w)
    } else {
        _linear_index(i, j, w)
    }
}

const fn _linear_index(i: usize, j: usize, w: usize) -> usize {
    i + j * w - j * (j + 3) / 2 - 1
}

fn matrix_index(k: usize, w: usize) -> (usize, usize) {
    assert!(k < w * (w - 1) / 2);
    let mut j = k / (w-1);
    while _linear_index(j+2, j+1, w) <= k { j += 1; }
    (k - _linear_index(j+1, j, w) + j + 1, j)
}

fn flat_distance<A>(matrix: &Array2<A>) -> Array1<A> where A: Clone + Copy {
    assert_eq!(matrix.ndim(), 2);
    let w = matrix.ncols();
    let mut flat;
    unsafe {
        // SAFETY: An uninitialised array is used for maximum speed.
        // It is unsafe to read uninitialised values, but it is safe to write to them.
        // There is no reading of uninitialised values here, only assignment.
        // Thus, this block is safe as long as all uninitialised values are written to,
        // which is the case (uncomment the assert to check, if not convinced).
        flat = Array1::uninitialized((w * w - w)/2);
        let mut k = 0;
        for i in 0..(w-1) {
            let j = w - 1 - i;
            flat.slice_mut(s![k..(k+j)]).assign(&matrix.slice(s![i,(i+1)..w]));
            k += j;
        }
        //assert_eq!(k, (w * w - w)/2);
    }
    flat
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
    let mut order: Vec<(usize, usize)> = (0..(dists.ncols()-1)).map(|i| ((i+1)..dists.ncols()).zip(std::iter::repeat(i))).flatten().collect();
    order.sort_unstable_by(|(i, j), (k, l)| dists[[*i, *j]].partial_cmp(&dists[[*k, *l]]).unwrap());
    //TODO: Maybe throw error if any dist is not comparable (NaN)
    for (i, j) in order.into_iter() {
        let d = dists[[i, j]];
        dists[[i, j]] = A::INFINITE_VALUE;
        flood_fill.flood_fill(&dists);
        if flood_fill.sizes.len() > clusters {
            dists[[i, j]] = d;
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
    fn index_transformation1() {
        assert_eq!(linear_index(1, 0, 5), 0);
        assert_eq!(linear_index(2, 0, 5), 1);
        assert_eq!(linear_index(2, 1, 5), 4);
        assert_eq!(linear_index(3, 2, 5), 7);
        assert_eq!(linear_index(4, 3, 5), 9);
        assert_eq!(linear_index(0, 1, 5), 0);
        assert_eq!(linear_index(2, 3, 5), 7);
    }

    #[test]
    fn index_transformation2() {
        assert_eq!(matrix_index(0, 5), (1, 0));
        assert_eq!(matrix_index(1, 5), (2, 0));
        assert_eq!(matrix_index(4, 5), (2, 1));
        assert_eq!(matrix_index(7, 5), (3, 2));
        assert_eq!(matrix_index(9, 5), (4, 3));
    }

    #[test]
    fn rslc1() {
        let x = array![[0,4,3],[4,0,2],[3,2,0]];
        rslc_slow(&x, 3, 0);
    }

    #[test]
    fn flat() {
        let x: Array2<f32> = Array2::zeros((5, 5));
        let flat = flat_distance(&x);
        assert_eq!(flat, Array1::zeros(10));
    }
}
