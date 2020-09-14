use extendr_api::{export_function, RObj, list};
use ndarray::{ArrayView2};
use paste::paste;
use rslc::rslc;

//--------------------------------------
// R Interface
//--------------------------------------

// TODO: Since extendr currently does not support windows this has not yet been tested!

macro_rules! r_rslc {
    ($($type:ident),+) => { $( paste! {
        #[export_function]
        fn [<rslc_ $type>](distances: ArrayView2<$type>, clusters: i32, min_size: i32) -> RObj 
        {
            let clusters: usize = clusters.usize();
            let min_size: usize = min_size.usize();

            let dists = distances.as_array();
            let (clusters, outliers) = rslc(&dists, clusters, min_size);
            list!(clusters = clusters, outliers = outliers)
        }
    } )+ };
}

r_rslc!(f64, i32);
