// Flight model of the ball in C++: the planner's hot path.
//
// The tracker pushes 13 sigma points through this model a few thousand times per camera frame
// (filter prediction plus a one-second forecast). In NumPy that costs 20 to 30 ms per frame, more
// than the 16.5 ms frame period. The arithmetic here is identical to g1catch/ballistics.py, which
// stays as the reference implementation, and tests/test_flight_core.py holds the two to 1e-12.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <stdexcept>

namespace py = pybind11;

namespace {

constexpr int kStateDim = 6;  // position (3), velocity (3)

struct FlightParams {
  double gravity_m_s2;
  double drag_per_m;            // kd in v' = -kd |v| v
  double ball_radius_m;
  double restitution;           // normal speed kept through a floor bounce
  double tangential_retention;  // horizontal speed kept through a floor bounce
  bool floor;
};

inline void acceleration(const double* v, const FlightParams& p, double* a) {
  const double speed = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  a[0] = -p.drag_per_m * speed * v[0];
  a[1] = -p.drag_per_m * speed * v[1];
  a[2] = -p.gravity_m_s2 - p.drag_per_m * speed * v[2];
}

// One RK4 step of a single state, then an instantaneous floor bounce if it ended below the floor
// while moving down. Same ordering of operations as ballistics.step.
inline void step_state(double* x, double dt, const FlightParams& p) {
  double* pos = x;
  double* vel = x + 3;
  double k1[3], k2[3], k3[3], k4[3], tmp[3];
  acceleration(vel, p, k1);
  for (int i = 0; i < 3; ++i) tmp[i] = vel[i] + 0.5 * dt * k1[i];
  acceleration(tmp, p, k2);
  for (int i = 0; i < 3; ++i) tmp[i] = vel[i] + 0.5 * dt * k2[i];
  acceleration(tmp, p, k3);
  for (int i = 0; i < 3; ++i) tmp[i] = vel[i] + dt * k3[i];
  acceleration(tmp, p, k4);
  for (int i = 0; i < 3; ++i) {
    pos[i] += dt * (vel[i] + dt / 6.0 * (k1[i] + k2[i] + k3[i]));
    vel[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
  }
  if (p.floor && pos[2] < p.ball_radius_m && vel[2] < 0.0) {
    pos[2] = 2.0 * p.ball_radius_m - pos[2];
    vel[2] *= -p.restitution;
    vel[0] *= p.tangential_retention;
    vel[1] *= p.tangential_retention;
  }
}

inline int substeps_for(double dt, double max_substep) {
  const int n = static_cast<int>(std::ceil(dt / max_substep - 1e-9));
  return n < 1 ? 1 : n;
}

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

void check_states(const Array& states) {
  if (states.ndim() != 2 || states.shape(1) != kStateDim) {
    throw std::invalid_argument("states must have shape (N, 6)");
  }
}

// Advance each row of `states` by dt, in substeps no longer than max_substep.
Array propagate(const Array& states, double dt, const FlightParams& p, double max_substep) {
  check_states(states);
  const py::ssize_t n = states.shape(0);
  Array out({n, static_cast<py::ssize_t>(kStateDim)});
  const double* src = states.data();
  double* dst = out.mutable_data();
  const int steps = substeps_for(dt, max_substep);
  const double h = dt / steps;
  for (py::ssize_t i = 0; i < n; ++i) {
    double* x = dst + i * kStateDim;
    for (int j = 0; j < kStateDim; ++j) x[j] = src[i * kStateDim + j];
    for (int s = 0; s < steps; ++s) step_state(x, h, p);
  }
  return out;
}

// States after dt, 2 dt, ..., steps * dt. Output shape (steps, N, 6).
Array rollout(const Array& states, double dt, int steps, const FlightParams& p, double max_substep) {
  check_states(states);
  if (steps < 1) throw std::invalid_argument("steps must be at least 1");
  const py::ssize_t n = states.shape(0);
  Array out({static_cast<py::ssize_t>(steps), n, static_cast<py::ssize_t>(kStateDim)});
  const double* src = states.data();
  double* dst = out.mutable_data();
  const int sub = substeps_for(dt, max_substep);
  const double h = dt / sub;
  for (py::ssize_t i = 0; i < n; ++i) {
    double x[kStateDim];
    for (int j = 0; j < kStateDim; ++j) x[j] = src[i * kStateDim + j];
    for (int k = 0; k < steps; ++k) {
      for (int s = 0; s < sub; ++s) step_state(x, h, p);
      double* row = dst + (static_cast<py::ssize_t>(k) * n + i) * kStateDim;
      for (int j = 0; j < kStateDim; ++j) row[j] = x[j];
    }
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(_flight_core, m) {
  m.doc() = "Ball flight model: RK4 with quadratic drag and a floor bounce.";
  py::class_<FlightParams>(m, "FlightParams")
      .def(py::init([](double gravity, double drag, double radius, double restitution, double retention, bool floor) {
             return FlightParams{gravity, drag, radius, restitution, retention, floor};
           }),
           py::arg("gravity_m_s2"), py::arg("drag_per_m"), py::arg("ball_radius_m"), py::arg("restitution"),
           py::arg("tangential_retention"), py::arg("floor") = true)
      .def_readwrite("drag_per_m", &FlightParams::drag_per_m)
      .def_readwrite("floor", &FlightParams::floor);
  m.def("propagate", &propagate, py::arg("states"), py::arg("dt"), py::arg("params"), py::arg("max_substep"));
  m.def("rollout", &rollout, py::arg("states"), py::arg("dt"), py::arg("steps"), py::arg("params"),
        py::arg("max_substep"));
}
