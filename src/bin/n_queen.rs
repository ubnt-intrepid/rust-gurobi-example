extern crate gurobi;
extern crate ndarray;
#[macro_use]
extern crate itertools;

use std::env::args;
pub use gurobi::*;
pub use itertools::*;
pub use ndarray::prelude::*;

fn main() {
  let n: usize = args()
    .nth(1)
    .and_then(|p| p.parse().ok())
    .unwrap_or(8);

  let env = Env::new("nqueen.log").unwrap();
  let mut model = Model::new("nqueen", &env).unwrap();
  model.get_env_mut().set(param::LogToConsole, 0).unwrap();

  let mut x = Vec::with_capacity(n * n);
  for (r, c) in iproduct!((0..n), (0..n)) {
    let var = model.add_var(&format!("x_{{{},{}}}", r, c),
               Binary,
               0.0,
               0.0,
               1.0,
               &[],
               &[])
      .unwrap();
    x.push(var);
  }
  model.update().unwrap();

  let x = Array::from_shape_vec((n, n), x).unwrap();

  // c0: each row must have exactly `one` queen.
  for (r, sb) in x.axis_iter(Axis(0)).enumerate() {
    let lhs = sb.fold(LinExpr::new(), |expr, x| expr + x);
    model.add_constr(&format!("c0_{}", r), lhs, Equal, 1.0).unwrap();
  }

  // c1: each column must have exactly `one` queen.
  for (c, sb) in x.axis_iter(Axis(1)).enumerate() {
    let lhs = sb.fold(LinExpr::new(), |expr, x| expr + x);
    model.add_constr(&format!("c1_{}", c), lhs, Equal, 1.0).unwrap();
  }

  // 左斜め
  for rr in 0..(n - 1) {
    let lhs = (rr..n).zip(0..(n - rr)).fold(LinExpr::new(), |expr, ix| expr + &x[ix]);
    model.add_constr(&format!("c2_{}", rr), lhs, Less, 1.0).unwrap();
  }
  for cc in 1..(n - 1) {
    let lhs = (0..(n - cc)).zip(cc..n).fold(LinExpr::new(), |expr, ix| expr + &x[ix]);
    model.add_constr(&format!("c3_{}", cc), lhs, Less, 1.0).unwrap();
  }

  // 右斜め
  for rr in 1..n {
    let lhs = (0..rr + 1).rev().zip(0..rr + 1).fold(LinExpr::new(), |expr, ix| expr + &x[ix]);
    model.add_constr(&format!("c4_{}", rr), lhs, Less, 1.0).unwrap();
  }
  for cc in 1..(n - 1) {
    let lhs = (cc..n).rev().zip(cc..n).fold(LinExpr::new(), |expr, ix| expr + &x[ix]);
    model.add_constr(&format!("c5_{}", cc), lhs, Less, 1.0).unwrap();
  }

  model.update().unwrap();
  model.write(&format!("{}_queen.lp", n)).unwrap();

  model.optimize().unwrap();
  model.write(&format!("{}_queen.sol", n)).unwrap();

  let sol = x.map(|x| x.get(&model, attr::X).unwrap());
  println!("{:?}", sol);
}
