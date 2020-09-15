
use std::collections::VecDeque;
use ndarray::{Array1, Array2, ArrayBase, Ix2, Data};
use num_traits::PrimInt;


//--------------------------------------
// Iterator over combinations
//--------------------------------------

/// An iterator that produces all unique sets of two integers up to a maximum size
/// 
/// # Examples
/// ```
/// use rslc::Combinations;
/// let combn: Vec<(u32, u32)> = Combinations::iter(4).collect();
/// assert_eq!(combn, vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
/// ```
pub struct Combinations<A> where A: PrimInt {
    n: A,
    a: A,
    b: A
}


impl<A> Iterator for Combinations<A> where A: PrimInt {
    type Item = (A, A);

    fn next(&mut self) -> Option<(A, A)> {
        self.b = self.b + A::one();
        if self.b >= self.n {
            self.a = self.a + A::one();
            if self.a + A::one() >= self.n {
                return Option::None;
            }
            self.b = self.a + A::one();
           
        }
        Option::Some((self.a, self.b))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(n) = self.n.to_usize() {
            if n > 1 {
                let size = n * (n - 1usize) / 2;
                return (size, Some(size));
            }
        }
        (0, None)
    }
}

impl<A> ExactSizeIterator for Combinations<A> where A: PrimInt {}

impl<A> Combinations<A> where A: PrimInt {
    /// Start the iterator over all unique sets of two integers (up to a maximum size).
    /// Note that the maximum size is exclusive (in order to math array index rules).
    /// 
    /// # Examples
    /// ```
    /// use rslc::Combinations;
    /// let combn: Vec<(u32, u32)> = Combinations::iter(4).collect();
    /// assert_eq!(combn, vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
    /// ```
    pub fn iter(n: A) -> Self {
        Combinations{n, a:A::zero(), b:A::zero()}
    }
}


//--------------------------------------
// Trait for distances
//--------------------------------------

/// A trait that can be used for distances between nodes in a graph.
/// 
/// By default it is implemented for all basic numerical types:
///   f32, f64, isize, i16, i32, i64, i128, usize, u16, u32, u64, u128.
/// 
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
dm!(isize, i16, i32, i64, i128; >=, 0, -1, 0);
dm!(usize, u16, u32, u64, u128; !=, Self::MAX, Self::MAX, 0);


//--------------------------------------
// Flood fill to find clusters
//--------------------------------------

/// Floodfill a graph to find all disconnected subgraphs. This can be used,
/// e.g., for finding clusters. The graph is generally represented by a
/// adjacency / distance matrix. It is implemented via a struct instead of a
/// pure function to make it easier to reuse the datastructures if the flood
/// fill is run multiple times (saves some memory allocation).
///
/// # Example
/// ```
/// use ndarray::Array2;
/// use rslc::{Combinations, FloodFill};
/// let mut ff = FloodFill::new();
/// let distance_matrix: Array2<f32> = Array2::zeros((5, 5));
/// ff.flood_fill(&distance_matrix);
/// ```
pub struct FloodFill {
    current: usize,
    cache: VecDeque<usize>,
    visited: Array1<bool>,
    clusters: Array1<usize>,
    sizes: Vec<usize>,
}

impl FloodFill {
    /// Create a new (empty) FloodFill struct.
    ///
    /// # Example
    /// ```
    /// use ndarray::Array2;
    /// use rslc::{Combinations, FloodFill};
    /// let mut ff = FloodFill::new();
    /// let distance_matrix: Array2<f32> = Array2::zeros((5, 5));
    /// ff.flood_fill(&distance_matrix);
    /// ```
    pub fn new() -> Self {
        FloodFill {
            current: 0,
            cache: VecDeque::new(),
            visited: Array1::default(0),
            clusters: Array1::default(0),
            sizes: Vec::new(),
        }
    }

    /// Create a new FloodFill struct and immediately use it on a graph.
    /// The graph is represented by a adjacency / distance matrix.
    /// The results are stored in the returned struct.
    ///
    /// # Example
    /// ```
    /// use ndarray::Array2;
    /// use rslc::{Combinations, FloodFill};
    /// let distance_matrix: Array2<f32> = Array2::zeros((5, 5));
    /// let mut ff = FloodFill::start(&distance_matrix);
    /// ```
    pub fn start<A>(adjacency_matrix: &Array2<A>) -> Self
    where A: DistanceMeasure,
    {
        let mut ff = Self::new();
        ff.flood_fill(adjacency_matrix);
        ff
    }

    /// Create a new FloodFill struct and put all items into the same cluster.
    /// The graph is represented by a adjacency / distance matrix.
    /// The results are stored in the returned struct.
    ///
    /// # Example
    /// ```
    /// use ndarray::Array2;
    /// use rslc::{Combinations, FloodFill};
    /// let mut distance_matrix: Array2<f32> = Array2::zeros((5, 5));
    /// let mut ff = FloodFill::init(&distance_matrix);
    /// ff.split_at(3, 4, &mut distance_matrix);
    /// ```
    pub fn init<A>(adjacency_matrix: &Array2<A>) -> Self
    where A: DistanceMeasure,
    {
        let mut ff = Self::new();
        ff.clusters = Array1::zeros(adjacency_matrix.ncols());
        ff.sizes.push(adjacency_matrix.ncols());
        ff
    }

    /// Use a FloodFill struct to flood fill a graph.
    /// The graph is represented by a adjacency / distance matrix.
    /// The results are stored in the struct, which is returned for chaining.
    ///
    /// # Example
    /// ```
    /// use ndarray::Array2;
    /// use rslc::{Combinations, FloodFill};
    /// let mut ff = FloodFill::new();
    /// let distance_matrix: Array2<f32> = Array2::zeros((5, 5));
    /// ff.flood_fill(&distance_matrix);
    /// ```
    pub fn flood_fill<A>(&mut self, adjacency_matrix: &Array2<A>) -> &mut Self
    where A: DistanceMeasure,
    {
        if self.clusters.len() != adjacency_matrix.ncols() {
            self.clusters = Array1::zeros(adjacency_matrix.ncols());
            self.current = 0;
            self.sizes.clear();
        } else {
            self.current += 1;
        }
        let start = self.current;
        let mut current = start;
        while let Some((i, _)) = self
            .clusters
            .iter()
            .enumerate()
            .find(|(_, x)| **x < start)
        {
            self.cache.clear();
            self.clusters[i] = current;
            self.cache.push_back(i);
            let mut count = 1;
            while let Some(i) = self.cache.pop_front() {
                for (j, _) in adjacency_matrix
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
            self.sizes.resize(current + 1, 0);
            self.sizes[current] = count;
            current += 1;
        }
        self.current = current;
        self
    }

    //TODO documentation and tests
    pub fn split_at<A>(&mut self, start: usize, end: usize, adjacency_matrix: &mut Array2<A>) -> Option<(usize, usize)>
    where A: DistanceMeasure,
    {
        adjacency_matrix[[start, end]] = A::INFINITE_VALUE;
        adjacency_matrix[[end, start]] = A::INFINITE_VALUE;
        if self.clusters.len() != adjacency_matrix.ncols() {
            self.clusters = Array1::zeros(adjacency_matrix.ncols());
            self.sizes.clear();
            self.sizes.push(adjacency_matrix.ncols());
            self.current = 0;
        }
        if self.visited.len() != adjacency_matrix.ncols() {
            self.visited = Array1::default(adjacency_matrix.ncols());
        } else {
            self.visited.fill(false);
        }
        // Breadth first search for the new cluster
        self.cache.clear();
        self.cache.push_back(start);
        self.visited[start] = true;
        while let Some(current) = self.cache.pop_front() {
            for (i, (_, v)) in adjacency_matrix
                .outer_iter()
                .nth(current)
                .unwrap()
                .iter()
                .zip(self.visited.iter_mut())
                .enumerate()
                .filter(|(_, (d, v))| d.is_finite() & !**v)
            {
                // Early exit, since the cluster did not split
                if adjacency_matrix[[end, i]].is_finite() {
                    return None;
                }
                *v = true;
                self.cache.push_back(i);
            }
        }
        // Apply the new cluster
        self.current += 1;
        let curr = self.current;
        let count = self.clusters.iter_mut()
            .zip(self.visited.iter())
            .filter(|(_, v)| **v)
            .map(|(c, _)| *c = curr)
            .count();
        self.sizes.resize(self.current + 1, 0);
        self.sizes[self.current] = count;
        self.sizes[self.clusters[end]] -= count;
        Some((self.sizes[self.current], self.sizes[self.clusters[end]]))
    }

    //TODO documentation and tests
    pub fn join_at<A>(&mut self, start: usize, end: usize, adjacency_matrix: &mut Array2<A>) -> &mut Self
    where A: DistanceMeasure + PartialEq,
    {
        let cluster_start = self.clusters[start];
        let cluster_end = self.clusters[end];
        if cluster_start != cluster_end {
            for c in &mut self.clusters {
                if *c == cluster_start {
                    *c = cluster_end;
                }
            }
            self.sizes[cluster_end] += self.sizes[cluster_start];
            self.sizes[cluster_start] = 0;
            if cluster_start == self.current {
                self.sizes.resize(self.current, 0);
                self.current -= 1;
            }
            if adjacency_matrix[[start, end]] == A::INFINITE_VALUE {
                adjacency_matrix[[start, end]] = A::MINIMUM_VALUE;
                adjacency_matrix[[end, start]] = A::MINIMUM_VALUE;
            }
        }
        self
    }

    //TODO documentation and tests
    pub fn get_clusters(&self) -> &Array1<usize> {
        &self.clusters
    }

    //TODO documentation and tests
    pub fn get_sizes(&self) -> &Vec<usize> {
        &self.sizes
    }

    //TODO documentation and tests
    pub fn get_num_clusters(&self) -> usize {
        self.sizes.iter().filter(|u| **u > 0).count()
    }

    //TODO documentation and tests
    pub fn clean_cluster_indices(&mut self) -> &mut Self {
        let mut order: Vec<usize> = (0..self.sizes.len()).collect();
        order.sort_unstable_by_key(|i| usize::MAX - self.sizes[*i]);
        let mut repl = vec![0; order.len()];
        for (i, o) in order.iter_mut().enumerate() {
            repl[*o] = i;
            *o = self.sizes[*o]
        }
        for c in &mut self.clusters {
            *c = repl[*c];
        }
        for (s, o) in self.sizes.iter_mut().zip(order.iter()) {
            *s = *o;
        }
        self.current = self.get_num_clusters() - 1;
        self.sizes.resize(self.current + 1, 0);
        self
    }
}


//--------------------------------------
// Robust Single Linkage Clustering
//--------------------------------------

/// TODO documentation
pub fn rslc<D, E>(distances: &ArrayBase<D, Ix2>, clusters: usize, min_size: usize) -> (Array1<usize>, Array1<bool>)
where 
    D: Data<Elem = E>,
    E: DistanceMeasure + Copy + PartialOrd,
{
    let mut dists = distances.to_owned();
    let mut outliers = Array1::default(distances.ncols());
    let mut flood_fill = FloodFill::init(&dists);
    let mut order: Vec<(usize, usize)> = Combinations::iter(dists.ncols()).collect();
    order.sort_unstable_by(|a, b| dists[*b].partial_cmp(&dists[*a]).unwrap());
    for (i, j) in order.into_iter() {
        if let Some((size_i, size_j)) = flood_fill.split_at(i, j, &mut dists) {
            // Check for outliers and enough clusters
            if size_i < min_size {
                let cls = flood_fill.get_clusters();
                let cl_i = cls[i];
                for (o, _) in outliers.iter_mut().zip(cls.iter()).filter(|(_, c)| **c == cl_i) {
                    *o = true;
                }
                flood_fill.join_at(i, j, &mut dists);
            } else if size_j < min_size {
                let cls = flood_fill.get_clusters();
                let cl_j = cls[j];
                for (o, _) in outliers.iter_mut().zip(cls.iter()).filter(|(_, c)| **c == cl_j) {
                    *o = true;
                }
                flood_fill.join_at(j, i, &mut dists);
            } else if flood_fill.get_num_clusters() == clusters {
                break;
            }
        }
    }
    flood_fill.clean_cluster_indices();
    (flood_fill.clusters, outliers)
}


//--------------------------------------
// Tests
//--------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array};

    #[test]
    fn combn() {
        let combs: Vec<(u32, u32)> = Combinations::iter(5).collect();
        assert_eq!(combs, vec![(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),]);
    }

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
        let a = clusters[0];
        let b = clusters[3];
        let c = clusters[6];
        assert_eq!(clusters, array![a,a,a, b,b,b, c,c,c,c,c,c, c,c]);
        assert!(c < a);
        assert!(c < b);
        assert_eq!(outliers, array![false,false,false, false,false,false, false,false,false,false,false,false, true,true]);
    }

}
