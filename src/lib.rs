use ndarray::{ArrayBase, Data, Ix2, Ix1, s, DataOwned, DataMut, Array1};
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

fn flat_distance<A, S1, S2>(matrix: &ArrayBase<S1, Ix2>) -> ArrayBase<S2, Ix1> where S1: Data<Elem = A>, S2: DataMut<Elem = A> + DataOwned, A: Clone + Copy {
    assert_eq!(matrix.ndim(), 2);
    let w = matrix.ncols();
    let mut flat;
    unsafe {
        // SAFETY: An uninitialised array is used for maximum speed.
        // It is unsafe to read uninitialised values, but it is safe to write to them.
        // There is no reading of uninitialised values here, only assignment.
        // Thus, this block is safe as long as all uninitialised values are written to,
        // which is the case (uncomment the assert to check, if not convinced).
        flat = ArrayBase::uninitialized((w * w - w)/2);
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

struct FloodFill<'a, S> where S: Data<Elem = usize> {
    start: usize,
    current: usize,
    cache: VecDeque<usize>,
    clusters: Array1<usize>,
    sizes: Vec<(usize, usize)>,
    distances: &'a ArrayBase<S, Ix2>,

}

impl<'a, S> FloodFill<'a, S> where S: Data<Elem = usize> {
    fn new(distances: &'a ArrayBase<S, Ix2>) -> Self {
        FloodFill {
            start: 0,
            current: 0,
            cache: VecDeque::with_capacity(distances.ncols()),
            clusters: Array1::zeros(distances.ncols()),
            sizes: vec!(),
            distances
        }
    }

    fn flood_fill(self: &mut Self) {
        self.start = self.current + 1;
        let mut current = self.start;
        self.sizes.clear();
        while let Some((i, _)) = self.clusters.iter().enumerate().find(|(i, x)| **x < self.start) {
            self.cache.clear();
            self.clusters[i] = current;
            self.cache.push_back(i);
            let mut count = 1;
            while let Some(i) = self.cache.pop_front() {
                for (j, _) in self.distances.outer_iter().nth(i).unwrap().iter().enumerate().filter(|(_, y)| **y < current) {
                    self.clusters[j] = self.current;
                    self.cache.push_back(j);
                    count += 1;
                }
            }
            self.sizes.push((current, count));
            current += 1;
        }
        self.current = current;
    }
}

fn partial_flood_fill<S1, S2>(dists: &ArrayBase<S1, Ix2>, start: usize, current: usize, clusters: &mut ArrayBase<S2, Ix1>) -> usize where S1: Data, S2: DataMut<Elem = usize> {
    match clusters.iter().enumerate().find(|(i, x)| **x < start){
        Some((i, _)) => {
            let mut i = i;
            let mut min = i;
            loop {

            }
            0
        },
        None => 0
    }
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
    use ndarray::{Array, OwnedRepr};
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
    fn flat() {
        let x: ArrayBase<OwnedRepr<f32>, Ix2> = Array::zeros((5, 5));
        let flat: ArrayBase<OwnedRepr<f32>, Ix1> = flat_distance(&x);
        assert_eq!(flat, Array::zeros(10));
    }
}
