
use numpy::{IntoPyArray, PyReadonlyArray2, PyArray1};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python, pyfunction};
use pyo3::wrap_pyfunction;
use paste::paste;
use rslc::rslc;

//--------------------------------------
// Python Interface for RSLC
//--------------------------------------

macro_rules! py_rslc1 {
    ($($type:ident),+) => { $( paste! {
        #[pyfunction]
        #[text_signature = "(distance_matrix, num_clusters, min_size, /)"]
        fn [<rslc_ $type>](py: Python<'_>, distances: PyReadonlyArray2<$type>, clusters: usize, min_size: usize) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<bool>>)> 
        {
            let dists = distances.as_array();
            let (clusters, outliers) = rslc(&dists, clusters, min_size);
            Ok((clusters.into_pyarray(py).to_owned(), outliers.into_pyarray(py).to_owned()))
        }
    } )+ };
}

macro_rules! py_rslc2 {
    ($m:ident, $($type:ident),+) => {$( paste! {
        $m.add_wrapped(wrap_pyfunction!([<rslc_ $type>])).unwrap();
    })+};
}

py_rslc1!(f32, f64, i16, i32, i64, u16, u32, u64);

#[pymodule(rslc)]
fn py_rslc(_py: Python<'_>, m: &PyModule) -> PyResult<()>
{
    py_rslc2!(m, f32, f64, i16, i32, i64, u16, u32, u64);

    Ok(())
}
