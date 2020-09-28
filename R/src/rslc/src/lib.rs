use ndarray::ArrayView;
use rslc::rslc;
use std::os::raw::{c_double, c_int};
use std::slice;

//--------------------------------------
// R Interface
//--------------------------------------

#[no_mangle]
pub unsafe extern "C" fn cluster(
    dists: *const c_double,
    clusters: *mut c_int,
    outliers: *mut c_int,
    items: c_int,
    num_clusters: c_int,
    min_size: c_int,
) {
    assert!(!dists.is_null(), "Null pointer for distances");
    assert!(!clusters.is_null(), "Null pointer for clusters");
    assert!(!outliers.is_null(), "Null pointer for outliers");
    assert!(items > 0, "The number of items must be greater than zero");
    assert!(
        num_clusters > 0,
        "The number of clusters must be greater than zero"
    );
    assert!(min_size >= 0, "The minimum size must be positive");

    let items = items as usize;
    let num_clusters = num_clusters as usize;
    let min_size = min_size as usize;
    let dists = slice::from_raw_parts(dists, items * items);
    match ArrayView::from_shape((items, items), dists) {
        Ok(dists) => {
            let clusters = slice::from_raw_parts_mut(clusters, items);
            let outliers = slice::from_raw_parts_mut(outliers, items);

            let (c, o) = rslc(&dists, num_clusters, min_size);

            for (out, res) in clusters.iter_mut().zip(c.iter()) {
                *out = *res as c_int + 1;
            }
            for (out, res) in outliers.iter_mut().zip(o.iter()) {
                *out = if *res { 1 } else { 0 };
            }
        }
        Err(e) => {
            println!("{}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_c_interface_test() {
        let dists: [c_double; 9] = [0.0, 4.0, 3.0, 4.0, 0.0, 2.0, 3.0, 2.0, 0.0];
        let mut cl: [c_int; 3] = [0, 0, 0];
        let mut out: [c_int; 3] = [1, 1, 1];

        unsafe {
            cluster(
                &dists as *const c_double,
                &mut cl[0] as *mut c_int,
                &mut out[0] as *mut c_int,
                3,
                2,
                1,
            );
        }
        assert_eq!(cl[0], 2);
        assert_eq!(cl[1], 1);
        assert_eq!(cl[2], 1);
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 0);
    }
}
