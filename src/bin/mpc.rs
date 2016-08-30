extern crate gurobi;
use gurobi::*;

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
  let solve_mip = |x_t: f64, t: usize| -> Result<Solution> {
    let horizon = 10;
    let mut model = try!(env.new_model(&format!("mip_{}", t)));

    Ok(Solution {
      status: Status::Infeasible,
      u: vec![0.0; horizon],
      x: vec![0.0; horizon],
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
    let x_t = x_t + u_t + 0.1;
    state.push(x_t);
  }
}
