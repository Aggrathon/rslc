//! # Robust Single-Linkage Clustering
//!
//! A variant of hierarchical clustering with a single-linkage merging function.
//! RSLC restricts the minimum size of the clusters, which makes it more robust
//! to outliers (and can even detect some of them). The RSLC algorithm is
//! inspired by the "Reverse-delete algorithm" for minimum spanning trees.

use std::collections::VecDeque;
use ndarray::{Array1, Array2, ArrayBase, Ix2, Data};
use num_traits::PrimInt;


/// takes two values and returns a sorted tuple
///
/// # Example
///
/// ```
/// use rslc::sort_tuple;
/// let (min1, max1) = sort_tuple(3, 8);
/// let (min2, max2) = sort_tuple(8, 3);
/// assert_eq!(min1, min2);
/// assert_eq!(max1, max2);
/// ```
pub fn sort_tuple<A>(a: A, b: A) -> (A, A) where A: PartialOrd {
    if a > b {
        (b, a)
    } else {
        (a, b)
    }
}

//--------------------------------------
// Iterator over combinations
//--------------------------------------

/// An iterator that produces all unique sets of two integers up to a maximum size
///
/// # Example
///
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
    /// # Arguments
    ///
    /// * `n` - the maximum size
    ///
    /// # Returns
    ///
    /// * `self` - a new iterator over all unique sets of size two with integers > 0 and < n
    ///
    /// # Example
    ///
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
// Flood fill to find clusters
//--------------------------------------

/// Maintain an undirected unweighted graph with clusters where edges can be
/// added and removed, changing the cluster memberships.
///
/// # Example
///
/// ```
/// use rslc::GraphClusters;
/// let mut cl = GraphClusters::new(5);
/// cl.disconnect(1, 2);
/// assert_eq!(cl.num_clusters(), 1)
/// ```
pub struct GraphClusters {
    current: usize,
    cache: VecDeque<usize>,
    visited: Array1<bool>,
    adjacency: Array2<bool>,
    clusters: Array1<usize>,
    sizes: Vec<usize>,
}

impl GraphClusters {
    /// Create a new GraphClusters struct where the graph is fully connected.
    ///
    /// # Arguments
    ///
    /// * `num_nodes` - the number of nodes in the graph
    ///
    /// # Return
    ///
    /// * `self` - a new GraphClusters struct
    ///
    /// # Example
    ///
    /// ```
    /// use rslc::GraphClusters;
    /// let mut cl = GraphClusters::new(5);
    /// assert_eq!(cl.num_clusters(), 1)
    /// ```
    pub fn new(num_nodes: usize) -> Self {
        let mut adj;
        unsafe {
            // Safety: the whole array is filled, so there are no uninitialised values
            adj = Array2::uninitialized((num_nodes, num_nodes));
            adj.fill(true);
        }
        let mut size = Vec::new();
        size.push(num_nodes);
        GraphClusters {
            current: 0,
            cache: VecDeque::new(),
            visited: Array1::default(num_nodes),
            clusters: Array1::default(num_nodes),
            adjacency: adj,
            sizes: size,
        }
    }

    /// Create a new GraphClusters struct where the graph is without any edges.
    ///
    /// # Arguments
    ///
    /// * `num_nodes` - the number of nodes in the graph
    ///
    /// # Return
    ///
    /// * `self` - a new GraphClusters struct
    ///
    /// # Example
    ///
    /// ```
    /// use rslc::GraphClusters;
    /// let mut cl = GraphClusters::new_split(5);
    /// cl.connect(1, 2);
    /// assert_eq!(cl.num_clusters(), 4)
    /// ```
    pub fn new_split(num_nodes: usize) -> Self {
        GraphClusters {
            current: 0,
            cache: VecDeque::new(),
            visited: Array1::default(num_nodes),
            clusters: (0..num_nodes).collect(),
            adjacency: Array2::default((num_nodes, num_nodes)),
            sizes: vec![1; num_nodes],
        }
    }

    /// Remove an edge from the graph
    ///
    /// # Arguments
    ///
    /// * `start` - one end of the edge (a node index)
    /// * `end` -  the other end of the edge (another node index)
    ///
    /// # Return
    ///
    /// Returns `None` if the edge is not removable (outside the graph or not
    /// existing) or if the removal didn't cause a change in the cluster
    /// structure.
    ///
    /// Returns `Some((usize, usize))` if a new cluster is created. The `Some`
    /// contains a tuple of the sizes of the new and old clusters.
    ///
    /// # Example
    ///
    /// ```
    /// use rslc::GraphClusters;
    /// let mut cl = GraphClusters::new(5);
    /// cl.disconnect(1, 0);
    /// cl.disconnect(1, 2);
    /// cl.disconnect(1, 3);
    /// cl.disconnect(1, 4);
    /// assert_eq!(cl.num_clusters(), 2)
    /// ```
    pub fn disconnect(&mut self, start: usize, end: usize) -> Option<(usize, usize)>
    {
        if start == end ||
            start >= self.clusters.len() ||
            end >= self.clusters.len() ||
            self.adjacency[[start, end]] == false {
            return  None;
        }
        self.adjacency[[start, end]] = false;
        self.adjacency[[end, start]] = false;
        self.visited.fill(false);
        // Breadth first search for the new cluster
        self.cache.clear();
        self.cache.push_back(start);
        self.visited[start] = true;
        while let Some(current) = self.cache.pop_front() {
            for (i, (_, v)) in self.adjacency
                .outer_iter()
                .nth(current)
                .unwrap()
                .iter()
                .zip(self.visited.iter_mut())
                .enumerate()
                .filter(|(_, (d, v))| **d & !**v)
            {
                // Early exit, since the cluster did not split
                if self.adjacency[[end, i]] {
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
        Some((count, self.sizes[self.clusters[end]]))
    }

    /// Adds an edge to the graph
    ///
    /// # Arguments
    ///
    /// * `start` - one end of the edge (a node index)
    /// * `end` -  the other end of the edge (another node index)
    ///
    /// # Return
    ///
    /// * `self` - for chaining
    ///
    /// # Example
    ///
    /// ```
    /// use rslc::GraphClusters;
    /// let mut cl = GraphClusters::new_split(5);
    /// cl.connect(1, 2);
    /// assert_eq!(cl.num_clusters(), 4)
    /// ```
    pub fn connect(&mut self, start: usize, end: usize) -> &mut Self
    {
        if start == end ||
            start >= self.clusters.len() ||
            end >= self.clusters.len() ||
            self.adjacency[[start, end]] == true {
            return self;
        }
        self.adjacency[[start, end]] = true;
        self.adjacency[[end, start]] = true;
        let (cla, clb) = (self.clusters[start], self.clusters[end]);
        if cla == clb {
            return self;
        }
        let (cla, clb) = sort_tuple(cla, clb);
        for c in &mut self.clusters {
            if *c == clb {
                *c = cla;
            }
        }
        self.sizes[cla] += self.sizes[clb];
        self.sizes[clb] = 0;
        if clb == self.current {
            self.sizes.resize(self.current, 0);
            self.current -= 1;
        }
        self
    }

    /// Get a reference to the array of cluster (index) per node
    pub fn get_clusters(&self) -> &Array1<usize> {
        &self.clusters
    }

    // Get the number of clusters with at least one member
    pub fn num_clusters(&self) -> usize {
        self.sizes.iter().filter(|u| **u > 0).count()
    }

    /// Shift the cluster indices to start from zero, sort the ckusters after
    /// size (descending), and remove empty clusters.
    ///
    /// # Return
    ///
    /// * `self` - for chaining
    ///
    pub fn clean_cluster_indices(&mut self) -> &mut Self {
        let mut order: Vec<usize> = (0..self.sizes.len()).collect();
        order.sort_unstable_by_key(|i| usize::MAX - self.sizes[*i]);
        // Check if already in order
        if None == order.iter().zip(0..self.sizes.len()).filter(|(o, i)| *o != i).nth(0) {
            self.current = self.num_clusters() - 1;
            self.sizes.resize(self.current + 1, 0);
            return self;
        }
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
        self.current = self.num_clusters() - 1;
        self.sizes.resize(self.current + 1, 0);
        self
    }
}


//--------------------------------------
// Robust Single Linkage Clustering
//--------------------------------------

/// **Robust Single-Linkage Clustering** is a variant of hierarchical clustering
/// with a single-linkage merging function. RSLC restricts the minimum size of
/// the clusters, which makes it more robust to outliers (and any cluster
/// smaller than that can be marked as outliers). The RSLC algorithm is inspired
/// by the "Reverse-delete algorithm" for minimum spanning trees.
///
/// # Arguments
///
/// * `distances` - A symmetric matrix of distances between items
/// * `num_clusters` - The number of clusters to find
/// * `min_size` - The minimum size of a cluster, used to find and avoid outliers
///
/// # Returns
///
/// * Vector of cluster indix per item (the indices starts from zero, ordered according to size)
/// * Binary vector of marked outliers
///
/// # Examples
/// ```
/// use rslc::rslc;
/// use ndarray::array;
/// let x = array![[0, 4, 3, 5, 1], [4, 0, 1, 5, 3], [3, 1, 0, 5, 2], [5, 5, 5, 0, 4], [1, 3, 2, 4, 0]];
/// let (clusters, outliers) = rslc(&x, 2, 2);
/// assert_eq!(array![false, false, false, true, false], outliers);
/// ```
pub fn rslc<D, E>(distances: &ArrayBase<D, Ix2>, num_clusters: usize, min_size: usize) -> (Array1<usize>, Array1<bool>)
where
    D: Data<Elem = E>,
    E: PartialOrd,
{
    let mut outliers = Array1::default(distances.ncols());
    let mut clusters = GraphClusters::new(distances.ncols());
    let mut order: Vec<(usize, usize)> = Combinations::iter(distances.ncols()).collect();
    order.sort_unstable_by(|a, b| distances[*b].partial_cmp(&distances[*a]).unwrap());
    for (i, j) in order.into_iter() {
        if let Some((size_i, size_j)) = clusters.disconnect(i, j) {
            // Check for outliers and enough clusters
            if size_i < min_size {
                let cls = clusters.get_clusters();
                let cl_i = cls[i];
                for (o, _) in outliers.iter_mut().zip(cls.iter()).filter(|(_, c)| **c == cl_i) {
                    *o = true;
                }
                clusters.connect(i, j);
            } else if size_j < min_size {
                let cls = clusters.get_clusters();
                let cl_j = cls[j];
                for (o, _) in outliers.iter_mut().zip(cls.iter()).filter(|(_, c)| **c == cl_j) {
                    *o = true;
                }
                clusters.connect(j, i);
            } else if clusters.num_clusters() == num_clusters {
                break;
            }
        }
    }
    dbg!(&clusters.clusters);
    clusters.clean_cluster_indices();
    dbg!(&clusters.clusters);
    (clusters.clusters, outliers)
}

// TODO: Try a bottom up approach where clusters are merged until k clusters are
// found, then if any of the clusters is too small it is marked as outlier and
// clusters are split until k non-outlier clusters are found. The difference to
// the results from the procedure above is that here outliers are ignored
// completely, where above they are assigned to the nearest cluster.


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

    #[test]
    fn graph_cluster_test() {
        let mut cl = GraphClusters::new_split(5);
        cl.connect(1, 2);
        assert_eq!(cl.num_clusters(), 4);
        cl.connect(3, 2);
        assert_eq!(cl.num_clusters(), 3);
        cl.connect(4, 0);
        assert_eq!(cl.num_clusters(), 2);
        cl.connect(4, 2);
        assert_eq!(cl.num_clusters(), 1);
        cl.disconnect(3, 2);
        assert_eq!(cl.num_clusters(), 2);
        cl.disconnect(0, 4);
        assert_eq!(cl.num_clusters(), 3);
        cl.disconnect(0, 2);
        assert_eq!(cl.num_clusters(), 3);
        cl.connect(0, 3);
        cl.clean_cluster_indices();
        assert_eq!(cl.get_clusters()[2], 0);
        assert_eq!(cl.get_clusters()[0], 1);
        cl = GraphClusters::new(5);
        assert_eq!(cl.num_clusters(), 1);
    }

}
