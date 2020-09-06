use ndarray::{ArrayBase, Data, Ix2, Ix1, s, DataOwned, DataMut};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::RuntimeError;
use pyo3::prelude::{pymodule, Py, PyErr, PyModule, PyResult, Python};
use std::fmt::Display;


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
    let mut j = k / w;
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
