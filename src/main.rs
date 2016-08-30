extern crate gurobi;
extern crate ndarray;
extern crate itertools;

use std::env::args;
use gurobi::{attr, param, Model, Var, LinExpr, Proxy, Binary, Env, Equal, Less};
use ndarray::prelude::*;
use itertools::*;

type Matrix<T> = Array<T, (usize, usize)>;
type Result<T> = std::result::Result<T, self::Error>;

#[derive(Debug)]
enum Error {
  Gurobi(gurobi::Error),
  ShapeError(ndarray::ShapeError),
  Other(String),
}

impl From<gurobi::Error> for Error {
  fn from(err: gurobi::Error) -> Error { Error::Gurobi(err) }
}

impl From<ndarray::ShapeError> for Error {
  fn from(err: ndarray::ShapeError) -> Error { Error::ShapeError(err) }
}

impl From<&'static str> for Error {
  fn from(err: &'static str) -> Error { Error::Other(err.to_owned()) }
}


trait IntoOk<T, E> {
  fn into_ok<U: From<T>>(self) -> std::result::Result<U, E>;
}

impl<T, E> IntoOk<T, E> for std::result::Result<T, E> {
  fn into_ok<U: From<T>>(self) -> std::result::Result<U, E> {
    self.map(|e| e.into())
  }
}

trait IntoErr<T, E> {
  fn into_err<Err: From<E>>(self) -> std::result::Result<T, Err>;
}

impl<T, E> IntoErr<T, E> for std::result::Result<T, E> {
  fn into_err<Err: From<E>>(self) -> std::result::Result<T, Err> {
    self.map_err(|e| e.into())
  }
}


fn make_matrix_variable(model: &mut Model,
                        rows: usize,
                        cols: usize,
                        name: &str)
                        -> Result<Matrix<Var>> {
  let mut vars = Vec::with_capacity(rows * cols);
  for (r, c) in (0..rows).cartesian_product(0..cols) {
    let var = try!(model.add_var(&format!("{}_{{{},{}}}", name, r, c), Binary));
    vars.push(var);
  }
  let vars = try!(Array::from_shape_vec((rows, cols), vars));
  Ok(vars)
}

fn get_solution_matrix(model: &Model, rows: usize, cols: usize) -> Result<Matrix<f64>> {
  let mut sol = Vec::with_capacity(rows * cols);
  for v in model.get_vars() {
    let x = try!(v.get(&model, attr::X));
    sol.push(x);
  }
  Array::from_shape_vec((rows, cols), sol).into_err()
}


fn n_queen(env: &Env, n: usize) -> Result<Matrix<f64>> {
  let mut model = try!(env.new_model("nqueen"));

  let x = try!(make_matrix_variable(&mut model, n, n, "x"));
  try!(model.update());

  // c0: each row must have exactly `one` queen.
  for (r, sb) in x.axis_iter(Axis(0)).enumerate() {
    let lhs = sb.fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c0_{}", r), lhs, Equal, 1.0));
  }

  // c1: each column must have exactly `one` queen.
  for (c, sb) in x.axis_iter(Axis(1)).enumerate() {
    let lhs = sb.fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c1_{}", c), lhs, Equal, 1.0));
  }

  // 左斜め
  for rr in 0..(n - 1) {
    let lhs = (rr..n).zip(0..(n - rr)).map(|ix| &x[ix]).fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c2_{}", rr), lhs, Less, 1.0));
  }
  for cc in 1..(n - 1) {
    let lhs = (0..(n - cc)).zip(cc..n).map(|ix| &x[ix]).fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c3_{}", cc), lhs, Less, 1.0));
  }

  // 右斜め
  for rr in 1..n {
    let lhs =
      (0..rr + 1).rev().zip(0..rr + 1).map(|ix| &x[ix]).fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c4_{}", rr), lhs, Less, 1.0));
  }
  for cc in 1..(n - 1) {
    let lhs = (cc..n).rev().zip(cc..n).map(|ix| &x[ix]).fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c5_{}", cc), lhs, Less, 1.0));
  }

  try!(model.write(&format!("{}_queen.lp", n)));
  try!(model.optimize());

  match try!(model.status()) {
    gurobi::Status::Optimal => {
      try!(model.write(&format!("{}_queen.sol", n)))
    }
    gurobi::Status::Infeasible => return Err("The model is infeasible.".into()),
    _ => return Err("Unknown error at optimization".into()),
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