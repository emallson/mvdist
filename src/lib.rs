#[macro_use]
extern crate lazy_static;
extern crate ndarray;
extern crate mvdist_sys;

use mvdist_sys::{mvcrit as sys_mvcrit, mvdist as sys_mvdist};
use ndarray::prelude::*;
use std::sync::Mutex;

#[derive(Clone, Debug, Copy)]
pub enum BoundType {
    Unbounded,
    Above,
    Below,
    Both,
}

impl Into<i32> for BoundType {
    fn into(self) -> i32 {
        match self {
            BoundType::Unbounded => -1,
            BoundType::Above => 0,
            BoundType::Below => 1,
            BoundType::Both => 2,
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct MVResult {
    pub value: f64,
    pub error: f64,
    pub nevals: i32,
    pub state: MVInform,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MVInform {
    Normal,
    PtLimitReached,
}

lazy_static! {
    static ref MVDIST_MUTEX: Mutex<()> = Mutex::new(());
}

fn column_ordered(ar: &Array2<f64>) -> Vec<f64> {
    ar.t().into_iter().cloned().collect()
}

/// Call the `mvdist` function from `mvdist-sys`. This function is *not* thread-safe. **Do not make
/// calls to it from multiple threads and expect better performance.** A mutex is used to ensure
/// this is the case.
pub fn mvdist(cov: &Array2<f64>,
              nu: i32,
              lb: &Array1<f64>,
              ub: &Array1<f64>,
              types: &Vec<BoundType>,
              constraints: &Array2<f64>,
              delta: &Array1<f64>,
              maxpts: i32,
              abseps: f64,
              releps: f64)
              -> Result<MVResult, String> {
    let shape = constraints.shape();
    let (m, n) = (shape[0] as i32, shape[1] as i32);
    let infin = types.iter().map(|&t| t.into()).collect::<Vec<i32>>();
    let guard = MVDIST_MUTEX.lock();
    let (error, value, nevals, inform) = sys_mvdist(n,
                                                    &column_ordered(cov),
                                                    nu,
                                                    m,
                                                    lb.as_slice().unwrap(),
                                                    &column_ordered(constraints),
                                                    ub.as_slice().unwrap(),
                                                    &infin,
                                                    delta.as_slice().unwrap(),
                                                    maxpts,
                                                    abseps,
                                                    releps);
    // I don't normally like to explicitly drop, but this ensures that the guard doesn't get elided
    // (which happens if let _ is used) and that I don't need #[allow(unused_variables)] to prevent
    // the warning.
    drop(guard);
    match inform {
            0 => Ok(MVInform::Normal),
            1 => Ok(MVInform::PtLimitReached),
            2 => Err(format!("Invalid choice of N")),
            3 => Err(format!("Covariance matrix not positive semidefinite")),
            x => Err(format!("Unknown error code {}", x)),
        }
        .and_then(|inf| {
            Ok(MVResult {
                error: error,
                value: value,
                nevals: nevals,
                state: inf,
            })
        })
}

pub fn mvcrit(cov: &Array2<f64>,
              nu: i32,
              lb: &Array1<f64>,
              ub: &Array1<f64>,
              types: &Vec<BoundType>,
              constraints: &Array2<f64>,
              alpha: f64,
              maxpts: i32,
              abseps: f64)
              -> Result<MVResult, String> {
    let shape = constraints.shape();
    let (m, n) = (shape[0] as i32, shape[1] as i32);
    let infin = types.iter().map(|&t| t.into()).collect::<Vec<i32>>();
    let guard = MVDIST_MUTEX.lock();
    let (error, value, nevals, inform) = sys_mvcrit(n,
                                                    &column_ordered(cov),
                                                    nu,
                                                    m,
                                                    lb.as_slice().unwrap(),
                                                    &column_ordered(constraints),
                                                    ub.as_slice().unwrap(),
                                                    &infin,
                                                    alpha,
                                                    maxpts,
                                                    abseps);
    // I don't normally like to explicitly drop, but this ensures that the guard doesn't get elided
    // (which happens if let _ is used) and that I don't need #[allow(unused_variables)] to prevent
    // the warning.
    drop(guard);
    match inform {
            0 => Ok(MVInform::Normal),
            1 => Ok(MVInform::PtLimitReached),
            2 => Err(format!("Invalid bounds given.")),
            x => Err(format!("Unknown error code {}", x)),
        }
        .and_then(|inf| {
            Ok(MVResult {
                error: error,
                value: value,
                nevals: nevals,
                state: inf,
            })
        })

}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn mvdist_works() {
        let cov = Array::eye(4);
        let con = arr2(&[[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0]]);
        let result = mvdist(&cov,
                            8,
                            &Array1::from_vec(vec![0.0; 5]),
                            &Array1::from_vec(vec![1.0; 5]),
                            &vec![BoundType::Both; 5],
                            &con,
                            &Array1::from_vec(vec![0.0; 5]),
                            100_000,
                            1e-5,
                            0.0)
            .unwrap();
        println!("{:?}", result);
        assert!(result.state == MVInform::Normal);
        assert!(result.nevals > 0 && result.nevals < 100_000);
        assert!((result.value - 0.001).abs() < 0.0001);
        assert!(result.error <= 1e-5);
    }

    #[test]
    fn mvdist_works2() {
        let a: Array2<f64> = arr2(&[[90.0, 60.0, 90.0],
                                    [90.0, 90.0, 30.0],
                                    [60.0, 60.0, 60.0],
                                    [60.0, 60.0, 90.0],
                                    [30.0, 30.0, 30.0]]);

        let con = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]);
        let centered = &a - &a.mean(Axis(0));
        let cov = &centered.t().dot(&centered) * (1.0 / (a.len_of(Axis(0)) as f64 - 1.0));

        mvdist(&cov,
               4,
               &Array::zeros((4,)),
               &Array1::from_vec(vec![100.0; 4]),
               &vec![BoundType::Both; 4],
               &con,
               &Array::zeros((4,)),
               100_000,
               1e-5,
               0.0)
            .unwrap();
    }
}
