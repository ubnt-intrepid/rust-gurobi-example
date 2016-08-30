#![allow(non_snake_case)]
#![allow(dead_code)]

extern crate gurobi;
#[macro_use]
extern crate itertools;
use gurobi::*;
use itertools::*;


fn main() {
  let mut env = Env::new("receding_horizon.log").unwrap();
  env.set(param::OutputFlag, 0).unwrap();
  let env = env;

  //
  struct Solution {
    status: Status,
    u: Vec<f64>,
    x: Vec<f64>,
  }

  fn add_var_series(model: &mut Model, name: &str, len: usize, start: isize) -> Result<Vec<Var>> {
    let mut vars = Vec::with_capacity(len);
    for i in start..((len as isize) - start) {
      let v = try!(model.add_var(&format!("{}_{}", name, i), Continuous(-INFINITY, INFINITY)));
      vars.push(v);
    }
    Ok(vars)
  }

  let solve_mip = |x_t: f64, t: usize| -> Result<Solution> {
    // control horizon.
    let horizon = 10;
    // stage/terminal cost
    let q = 100.0;
    let r = 0.42;
    let s = 0.01;

    let mut model = try!(env.new_model(&format!("mip_{}", t)));

    let u = try!(add_var_series(&mut model, "u", horizon, 0));
    let x = try!(add_var_series(&mut model, "x", horizon + 2, -1));
    try!(model.update());

    // u_k \in [-1.0, 1.0]
    for u in u.iter() {
      try!(u.set(&mut model, attr::LB, -1.0));
      try!(u.set(&mut model, attr::UB, 1.0));
    }

    // Initial state variable must equal to initial state.
    try!(model.add_constr("initial", 1.0 * &x[0], Equal, x_t));

    // remaining variables must satisfy the state equation at each step.
    for (k, (u_k, x_k, x_k1)) in Zip::new((u.iter(), x.iter(), x.iter().skip(1))).enumerate() {
      try!(model.add_constr(&format!("ss_{}", k),
                            x_k1 + (-0.9 * x_k) + (-1.0 * u_k),
                            Equal,
                            0.0));
    }

    // set objective function
    let expr = Zip::new((x.iter().skip(1), u.iter())).fold(QuadExpr::new(), |expr, (x, u)| {
      expr + (x * x) * q + (u * u) * r
    }) + try!(x.last().map(|x_T| (x_T * x_T) * s).ok_or(Error::InconsitentDims));
    try!(model.set_objective(expr, Minimize));

    // solve optimization.
    try!(model.optimize());
    match try!(model.status()) {
      Status::Optimal => (),
      status => {
        return Ok(Solution {
          status: status,
          u: vec![],
          x: vec![],
        })
      }
    }

    let mut sol_u = Vec::with_capacity(u.len());
    for u in u.into_iter() {
      let u = try!(u.get(&model, attr::X));
      sol_u.push(u);
    }

    let mut sol_x = Vec::with_capacity(x.len());
    for x in x.into_iter() {
      let x = try!(x.get(&model, attr::X));
      sol_x.push(x);
    }

    Ok(Solution {
      status: Status::Optimal,
      u: sol_u,
      x: sol_x,
    })
  };

  // configurations.
  let n_times = 100;
  let n_rt = 10;
  let x_0 = 1.0;

  // the buffers to contain the sequence of state/input.
  let mut state = Vec::with_capacity(n_times * n_rt);
  let mut input = Vec::with_capacity(n_times * n_rt);

  // store the initial state to buffer
  state.push(x_0);

  for t in 0..(n_times * n_rt) {
    let x_t = state.last().cloned().unwrap();

    if t % n_rt == 0 {
      // update MPC input.
      let sol = solve_mip(x_t, t).unwrap();
      let u = match sol.status {
        Status::Optimal => *sol.u.get(0).unwrap(),
        _ => {
          println!("step {}: cannot retrieve an optimal MIP solution", t);
          0.0
        }
      };
      input.push(u);
    }
    let u_t = input.last().cloned().unwrap();

    // update the value of actual state.
    let x_t = 0.99*x_t + u_t + 0.01;
    state.push(x_t);
  }

  println!("input = {:?}", input);
//  println!("state = {:?}", state);
}
