use ndarray::{s, Array1, Array2};
use num_traits::{PrimInt};


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
}

impl<A> Combinations<A> where A: PrimInt {
    pub fn iter(n: A) -> Self {
        Combinations{n, a:A::zero(), b:A::zero()}
    }
}



fn flat_index(i: usize, j: usize, width: usize) -> usize {
    assert!(i != j);
    assert!(j < width);
    assert!(i < width);
    if i < j { 
        _flat_index(j, i, width)
    } else {
        _flat_index(i, j, width)
    }
}

const fn _flat_index(i: usize, j: usize, w: usize) -> usize {
    i + j * w - j * (j + 3) / 2 - 1
}

fn matrix_index(flat_index: usize, width: usize) -> (usize, usize) {
    assert!(flat_index < width * (width - 1) / 2);
    let mut j = flat_index / (width-1);
    while _flat_index(j+2, j+1, width) <= flat_index { j += 1; }
    (flat_index - _flat_index(j+1, j, width) + j + 1, j)
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


#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use super::*;

    #[test]
    fn combn() {
        let combs: Vec<(u32, u32)> = Combinations::iter(5).collect();
        assert_eq!(combs, vec![(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),]);
    }

    #[test]
    fn index_transformation1() {
        assert_eq!(flat_index(1, 0, 5), 0);
        assert_eq!(flat_index(2, 0, 5), 1);
        assert_eq!(flat_index(2, 1, 5), 4);
        assert_eq!(flat_index(3, 2, 5), 7);
        assert_eq!(flat_index(4, 3, 5), 9);
        assert_eq!(flat_index(0, 1, 5), 0);
        assert_eq!(flat_index(2, 3, 5), 7);
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
        let x: Array2<f32> = Array2::zeros((5, 5));
        let flat = flat_distance(&x);
        assert_eq!(flat, Array1::zeros(10));
    }
}
