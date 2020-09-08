mod utils;

use ndarray::{Array1, Array2, ArrayBase, Ix2, Data};
use std::collections::VecDeque;
use utils::Combinations;

use numpy::{IntoPyArray, PyReadonlyArray2, PyArray1};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python, pyfunction};
use pyo3::wrap_pyfunction;
use paste::paste;


//--------------------------------------
// Trait for distances
//--------------------------------------

pub trait DistanceMeasure {
    fn is_finite(&self) -> bool;
    const INFINITE_VALUE: Self;
    const MINIMUM_VALUE: Self;
}

macro_rules! dm {
    ($($type:ident),+; $isf_tt:tt, $isf_v:expr, $inf:expr, $min:expr) => {$(
        impl DistanceMeasure for $type {
            fn is_finite(&self) -> bool { *self $isf_tt $isf_v }
            const INFINITE_VALUE: Self = $inf;
            const MINIMUM_VALUE: Self = $min;
        }
     )+};
}

dm!(f32, f64; >=, 0.0, -1.0, 0.0);
dm!(isize, i16,  i32, i64, i128; >=, 0, -1, 0);
dm!(usize, u16, u32, u64, u128; !=, Self::MAX, Self::MAX, 0);


//--------------------------------------
// Flood fill to find clusters
//--------------------------------------

pub struct FloodFill {
    start: usize,
    current: usize,
    cache: VecDeque<usize>,
    clusters: Array1<usize>,
    sizes: Vec<(usize, usize)>,
}

impl FloodFill {
    pub fn new() -> Self {
        FloodFill {
            start: 0,
            current: 0,
            cache: VecDeque::new(),
            clusters: Array1::zeros(0),
            sizes: Vec::new(),
        }
    }

    pub fn start<A>(distances: &Array2<A>) -> Self
    where A: DistanceMeasure,
    {
        let mut ff = Self::new();
        ff.flood_fill(distances);
        ff
    }

    pub fn flood_fill<A>(&mut self, distances: &Array2<A>)
    where A: DistanceMeasure,
    {
        if self.clusters.len() != distances.ncols() {
            self.clusters = Array1::zeros(distances.ncols());
        }
        self.start = self.current + 1;
        let mut current = self.start;
        self.sizes.clear();
        while let Some((i, _)) = self
            .clusters
            .iter()
            .enumerate()
            .find(|(_, x)| **x < self.start)
        {
            self.cache.clear();
            self.clusters[i] = current;
            self.cache.push_back(i);
            let mut count = 1;
            while let Some(i) = self.cache.pop_front() {
                for (j, _) in distances
                    .outer_iter()
                    .nth(i)
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter(|(_, y)| y.is_finite())
                {
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


//--------------------------------------
// Robust Single Linkage Clustering
//--------------------------------------

fn find_outliers(flood_fill: &FloodFill, outliers: &mut Array1<bool>, min_size: usize) -> bool {
    let mut new_outliers = false;
    for (i, _) in flood_fill.sizes.iter().filter(|(_, s)| *s < min_size) {
        for (o, _) in outliers
            .iter_mut()
            .zip(flood_fill.clusters.iter())
            .filter(|(_, c)| *c == i)
        {
            *o = true;
        }
        new_outliers = true;
    }
    new_outliers
}

fn breadth_first_search<A>(distances: &Array2<A>, start: usize, end: usize) -> bool
where A: DistanceMeasure,
{
    let mut queue = VecDeque::new();
    let mut visited = vec![false; distances.ncols()];
    queue.push_back(start);
    visited[start] = true;
    while let Some(current) = queue.pop_front() {
        for (i, (_, v)) in distances
            .outer_iter()
            .nth(current)
            .unwrap()
            .iter()
            .zip(visited.iter_mut())
            .filter(|(d, v)| d.is_finite() & !**v)
            .enumerate()
        {
            // Early exit, since this checks for possible path, not shortest
            if distances[[end, i]].is_finite() {
                return true;
            }
            *v = true;
            queue.push_back(i);
        }
    }
    false
}

pub fn rslc<D, E>(distances: &ArrayBase<D, Ix2>, clusters: usize, min_size: usize) -> (Array1<usize>, Array1<bool>)
where 
    D: Data<Elem = E>,
    E: DistanceMeasure + Copy + PartialOrd,
{
    let mut dists = distances.to_owned();
    let mut outliers = Array1::default(distances.ncols());
    let mut flood_fill = FloodFill::new();
    let mut order: Vec<(usize, usize)> = Combinations::iter(dists.ncols()).collect();
    order.sort_unstable_by(|a, b| dists[*b].partial_cmp(&dists[*a]).unwrap());
    for (i, j) in order.into_iter() {
        dists[[i, j]] = E::INFINITE_VALUE;
        dists[[j, i]] = E::INFINITE_VALUE;
        // if i and j are still connected no checks for new clusters are needed
        if !breadth_first_search(&dists, i, j) {
            flood_fill.flood_fill(&dists);
            if flood_fill.sizes.len() > clusters {
                dists[[i, j]] = E::MINIMUM_VALUE;
                dists[[j, i]] = E::MINIMUM_VALUE;
                flood_fill.flood_fill(&dists);
                break;
            }
            if find_outliers(&flood_fill, &mut outliers, min_size) {
                dists[[i, j]] = E::MINIMUM_VALUE;
            }
        }
    }
    (flood_fill.clusters - flood_fill.start, outliers)
}

//--------------------------------------
// Python Interface
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


//--------------------------------------
// Tests
//--------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array};

    #[test]
    fn rslc_basic() {
        let x = array![[0, 4, 3], [4, 0, 2], [3, 2, 0]];
        rslc(&x, 3, 0);
    }

    #[test]
    fn rslc_truth() {
        // This distance matrix describes a smiley-face with to outliers below the mouth.
        // The expected clusters are the eyes and the mouth.
        let x = array![
            [0,1,1, 4,4,4, 2,2,3,4,5,6, 5,6],
            [1,0,1, 4,4,4, 2,2,3,4,5,6, 5,6],
            [1,1,0, 4,4,4, 2,2,3,4,5,6, 5,6],

            [4,4,4, 0,1,1, 6,5,4,3,2,2, 6,5],
            [4,4,4, 1,0,1, 6,5,4,3,2,2, 6,5],
            [4,4,4, 1,1,0, 6,5,4,3,2,2, 6,5],

            [2,2,2 ,6,6,6, 0,1,2,3,4,5, 4,6],
            [2,2,2 ,5,5,5, 1,0,1,2,3,4, 3,5],
            [3,3,3 ,4,4,4, 2,1,0,1,2,3, 3,4],
            [4,4,4 ,3,3,3, 3,2,1,0,1,2, 4,3],
            [5,5,5 ,2,2,2, 4,3,2,1,0,1, 5,3],
            [6,6,6 ,2,2,2, 5,4,3,2,1,0, 6,4],

            [5,5,5 ,6,6,6, 4,3,3,4,5,6, 0,4],
            [6,6,6 ,5,5,5, 6,5,4,3,3,4, 4,0],];
        let (clusters, outliers) = rslc(&x, 3, 2);
        assert_eq!(outliers, array![false,false,false, false,false,false, false,false,false,false,false,false, true,true]);
        assert_eq!(clusters, array![0,0,0, 1,1,1, 2,2,2,2,2,2, 2,2]);
    }
}
