extern crate gurobi;
extern crate ndarray;

use std::env::args;
use gurobi::{attr, param, Model, Var, LinExpr, Proxy, Binary, Env, Equal, Less};
use ndarray::prelude::*;

type Matrix<T> = Array<T, (usize, usize)>;
type Result<T> = std::result::Result<T, String>;


fn make_matrix_variable(model: &mut Model,
                        rows: usize,
                        cols: usize,
                        name: &str)
                        -> Result<Matrix<Var>> {
  let mut vars = Vec::with_capacity(rows * cols);
  for (r, c) in (0..rows).flat_map(|r| (0..cols).map(|c| (r, c)).collect::<Vec<_>>()) {
    let var = try!(model.add_var(&format!("{}_{{{},{}}}", name, r, c), Binary)
      .map_err(|e| format!("{:?}", e)));
    vars.push(var);
  }
  let vars = try!(Array::from_shape_vec((rows, cols), vars).map_err(|e| e.to_string()));
  Ok(vars)
}

fn get_solution_matrix(model: &Model, rows: usize, cols: usize) -> Result<Matrix<f64>> {
  let mut sol = Vec::with_capacity(rows * cols);
  for v in model.get_vars() {
    let x = try!(v.get(&model, attr::X).map_err(|e| format!("{:?}", e)));
    sol.push(x);
  }
  Array::from_shape_vec((rows, cols), sol).map_err(|e| e.to_string())
}


fn n_queen(env: &Env, n: usize) -> Result<Matrix<f64>> {
  let mut model = try!(env.new_model("nqueen").map_err(|e| format!("{:?}", e)));

  let x = try!(make_matrix_variable(&mut model, n, n, "x"));
  try!(model.update().map_err(|e| format!("{:?}", e)));

  // c0: each row must have exactly `one` queen.
  for (r, sb) in x.axis_iter(Axis(0)).enumerate() {
    let lhs = sb.fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c0_{}", r), lhs, Equal, 1.0).map_err(|e| format!("{:?}", e)));
  }

  // c1: each column must have exactly `one` queen.
  for (c, sb) in x.axis_iter(Axis(1)).enumerate() {
    let lhs = sb.fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c1_{}", c), lhs, Equal, 1.0).map_err(|e| format!("{:?}", e)));
  }

  // 左斜め
  for rr in 0..(n - 1) {
    let lhs = (rr..n).zip(0..(n - rr)).map(|ix| &x[ix]).fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c2_{}", rr), lhs, Less, 1.0).map_err(|e| format!("{:?}", e)));
  }
  for cc in 1..(n - 1) {
    let lhs = (0..(n - cc)).zip(cc..n).map(|ix| &x[ix]).fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c3_{}", cc), lhs, Less, 1.0).map_err(|e| format!("{:?}", e)));
  }

  // 右斜め
  for rr in 1..n {
    let lhs =
      (0..rr + 1).rev().zip(0..rr + 1).map(|ix| &x[ix]).fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c4_{}", rr), lhs, Less, 1.0).map_err(|e| format!("{:?}", e)));
  }
  for cc in 1..(n - 1) {
    let lhs = (cc..n).rev().zip(cc..n).map(|ix| &x[ix]).fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c5_{}", cc), lhs, Less, 1.0).map_err(|e| format!("{:?}", e)));
  }

  try!(model.write(&format!("{}_queen.lp", n)).map_err(|e| format!("{:?}", e)));
  try!(model.optimize().map_err(|e| format!("{:?}", e)));

  match try!(model.status().map_err(|e| format!("{:?}", e))) {
    gurobi::Status::Optimal => {
      try!(model.write(&format!("{}_queen.sol", n)).map_err(|e| format!("{:?}", e)))
    }
    gurobi::Status::Infeasible => return Err("The model is infeasible.".to_owned()),
    _ => return Err("Unknown error at optimization".to_owned()),
  }
  get_solution_matrix(&model, n, n)
}

fn main() {
  let mut env = Env::new("nqueen.log").unwrap();
  env.set(param::LogToConsole, 0).unwrap();

  let n: usize = args()
    .nth(1)
    .and_then(|p| p.parse().ok())
    .unwrap_or(8);

  match n_queen(&env, n) {
    Ok(sol) => println!("solution is:\n{:?}", sol),
    Err(err) => println!("failed to solve model with error: {:?}", err),
  }
}
