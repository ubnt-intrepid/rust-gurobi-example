extern crate gurobi;
extern crate ndarray;
use gurobi::*;
use ndarray::prelude::*;

fn make_matrix_variable(model: &mut gurobi::Model, rows: usize, cols: usize, name: &str) -> Result<Vec<Var>> {
  let mut vars = Vec::with_capacity(rows * cols);
  for r in 0..rows {
    for c in 0..cols {
      let vname = format!("{}[{}][{}]", name, r, c);
      let var = try!(model.add_var(&vname, Binary));
      vars.push(var);
    }
  }
  Ok(vars)
}

fn n_queen(env: &Env, n: usize) -> Result<Array<f64, (usize, usize)>> {
  let mut model = try!(env.new_model("nqueen"));

  let x = try!(make_matrix_variable(&mut model, n, n, "x"));
  try!(model.update());
  let x = try!(Array::from_shape_vec((n, n), x).map_err(|_| gurobi::Error::InconsitentDims));

  for r in 0..n {
    let sb = x.subview(Axis(0), r);
    let lhs = sb.fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c0[{}]", r), lhs, Equal, 1.0));
  }

  for c in 0..n {
    let sb = x.subview(Axis(1), c);
    let lhs = sb.fold(LinExpr::new(), |expr, x| expr + x);
    try!(model.add_constr(&format!("c1[{}]", c), lhs, Equal, 1.0));
  }

  try!(model.optimize());

  let mut sol = Vec::with_capacity(n * n);
  for v in model.get_vars() {
    let x = try!(v.get(&model, attr::X));
    sol.push(x);
  }

  Array::from_shape_vec((n, n), sol).map_err(|_| gurobi::Error::InconsitentDims)
}


fn main() {
  let mut env = Env::new("nqueen.log").unwrap();
  env.set(param::LogToConsole, 0).unwrap();

  match n_queen(&env, 3) {
    Ok(sol) => println!("solution is:\n{:?}", sol),
    Err(err) => println!("failed to solve model: {:?}", err),
  }
}
