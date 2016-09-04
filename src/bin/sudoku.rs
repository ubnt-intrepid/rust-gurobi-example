extern crate gurobi;
extern crate ndarray;
#[macro_use]
extern crate itertools;

use std::ops::Add;
use std::env::args;
use std::fs::File;
use std::io::{BufReader, BufRead};

use gurobi::*;
use ndarray::prelude::*;

fn main() {
  let sd = 3;
  let n = sd * sd;

  let f = File::open(args().nth(1).unwrap()).unwrap();
  let reader = BufReader::new(f);

  let data: Vec<_> = reader.lines()
    .flat_map(|line| line.unwrap().trim().chars().collect::<Vec<_>>())
    .collect();

  let env = Env::new("sudoku.log").unwrap();
  let mut model = Model::new("sudoku", &env).unwrap();

  // Create 3-dimensional array of variables.
  let mut vars = Vec::with_capacity(n * n * n);
  for (i, j, v) in iproduct!((0..n), (0..n), (0..n)) {
    let vname = format!("G_{}_{}_{}", i, j, v);
    let v = model.add_var(&vname, Binary, 0.0, 0.0, 1.0, &[], &[]).unwrap();
    vars.push(v);
  }
  let vars = Array::from_shape_vec((n, n, n), vars).unwrap();

  model.update().unwrap();

  // Each cell must take one value.
  for (i, j) in iproduct!((0..n), (0..n)) {
    let expr = vars.axis_iter(Axis(2)).fold(LinExpr::new(), |expr, sb| expr + &sb[(i, j)]);
    model.add_constr(&format!("V_{}_{}", i, j), expr, Equal, 1.0).unwrap();
  }

  // Each value appears once per row.
  for (i, v) in iproduct!((0..n), (0..n)) {
    let expr = vars.axis_iter(Axis(1)).fold(LinExpr::new(), |expr, sb| expr + &sb[(i, v)]);
    model.add_constr(&format!("R_{}_{}", i, v), expr, Equal, 1.0).unwrap();
  }

  // Each value appears once per column.
  for (j, v) in iproduct!((0..n), (0..n)) {
    let expr = vars.axis_iter(Axis(0)).fold(LinExpr::new(), |expr, sb| expr + &sb[(j, v)]);
    model.add_constr(&format!("C_{}_{}", j, v), expr, Equal, 1.0).unwrap();
  }

  // Each value appears once per sub-grid
  for (i0, j0, v) in iproduct!((0..sd), (0..sd), (0..n)) {
    let expr = iproduct!((0..sd), (0..sd))
      .map(|(i1, j1)| &vars[(i0 * sd + i1, j0 * sd + j1, v)])
      .fold(LinExpr::new(), Add::add);

    model.add_constr(&format!("Sub_{}_{}_{}", v, i0, j0), expr, Equal, 1.0).unwrap();
  }

  // Fix variables associated with pre-specified cells.
  for ((i0, j0), val) in iproduct!((0..n), (0..n)).zip(data) {
    match val {
      v @ '1'...'9' => {
        let v = (v as usize) - 48 - 1; // 0-based
        vars[(i0, j0, v)].set(&mut model, attr::LB, 1.0).unwrap();
      }
      _ => (),
    }
  }

  // Optimize model
  model.optimize().unwrap();

  // Write model to file.
  model.write("sudoku.lp").unwrap();
  model.write("sudoku.sol").unwrap();

  println!("");
  for i in 0..n {
    for j in 0..n {
      let x: usize = vars.axis_iter(Axis(2))
        .map(|v| v[(i, j)].clone())
        .map(|v| v.get(&model, attr::X).unwrap() as usize)
        .enumerate()
        .map(|(i, x)| (i + 1) * x)
        .sum();
      print!("{}", x);
    }
    println!("");
  }
}
